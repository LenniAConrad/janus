//! JRBT evaluator: the E04 relation-biased transformer (a BT4-style transformer
//! whose attention is biased by explicit typed chess relations). Trained in
//! nn-research; beats a matched BT4 transformer at equal budget. Consumes the
//! LC0 112-plane input and emits 1858 LC0 policy + WDL + moves-left, so it
//! reuses the BT4 encoder + 1858→move mapping and is MCTS-only like BT4.
//!
//! Byte format ("JRBT", little-endian): magic, i32 version=1, then 10 i32 config
//! fields + 8 u8 config fields, i32 n_tensors, per-tensor `[i32 namelen, name,
//! i32 ndim, dims, f32 data]`, then `i32 pm_len` + `i32 policy_map[pm_len]`.
//! Forward matches nn-research `jrbt_reference.py` (numpy==torch to 1e-7).

// The forward pass mirrors the nn-research reference kernels one-for-one, so
// its local names stay the reference's short mathematical ones (`x`, `w`, `b`,
// `d`, `r`) and its configuration mirrors the file's boolean feature switches.
// Restyling either would break the line-by-line correspondence that validates
// this port, exactly as in `examples/hce_texel_tune.rs`.
// The float casts are the reference's own `float(...)` conversions over
// bounded board-sized counts, so the pedantic precision/truncation lints are
// silenced module-wide rather than at dozens of call sites.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::explicit_iter_loop,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::unused_self,
    clippy::struct_excessive_bools,
    clippy::unreadable_literal
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use janus_core::{Move, Position};

use crate::bt4::Bt4InputFormat;
use crate::bt4_encoding::{
    encode_lc0_fen_position_into, gather_legal_policy_logits, INPUT_VALUES, POLICY_SIZE,
};
use crate::evaluator::{Evaluator, PolicyValue};

const MAGIC: &[u8; 4] = b"JRBT";
const VERSION: i32 = 1;
const TOKENS: usize = 64;
const INPUT_CHANNELS: usize = 112;
const RAW_POLICY_SIZE: usize = 4_096 + 192;
/// Maximum accepted JRBT file size, including its tensor and policy payloads.
pub const MAX_MODEL_BYTES: usize = 512 * 1024 * 1024;
const MAX_NETWORK_WIDTH: usize = 4_096;
const MAX_BLOCKS: usize = 64;
const MAX_HEADS: usize = 128;
const MAX_FFN_WIDTH: usize = 16_384;
const MAX_RELATIONS: usize = 16;
const MAX_SMOLGEN_WIDTH: usize = 16_384;
const MAX_TENSORS: usize = 4_096;
const MAX_TENSOR_RANK: usize = 8;
const MAX_TENSOR_NAME_BYTES: usize = 256;
const MAX_TENSOR_AXIS: usize = 1_048_576;
const MAX_TENSOR_ELEMENTS: usize = MAX_MODEL_BYTES / std::mem::size_of::<f32>();

/// Bounded load/parse failure raised while admitting a JRBT weight file.
///
/// The payload is a human-readable reason; it never carries model bytes.
#[derive(Debug)]
pub struct JrbtError(
    /// Used for carrying the bounded human-readable failure reason.
    pub String,
);
impl std::fmt::Display for JrbtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JRBT load error: {}", self.0)
    }
}
impl std::error::Error for JrbtError {}

fn err<T>(m: impl Into<String>) -> Result<T, JrbtError> {
    Err(JrbtError(m.into()))
}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], JrbtError> {
        let end = self
            .off
            .checked_add(n)
            .ok_or_else(|| JrbtError("reader offset overflow".to_owned()))?;
        if end > self.b.len() {
            return err("unexpected EOF");
        }
        let s = &self.b[self.off..end];
        self.off = end;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32, JrbtError> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u8(&mut self) -> Result<u8, JrbtError> {
        Ok(self.take(1)?[0])
    }

    fn bounded_usize(
        &mut self,
        label: &str,
        minimum: usize,
        maximum: usize,
    ) -> Result<usize, JrbtError> {
        let raw = self.i32()?;
        let value =
            usize::try_from(raw).map_err(|_| JrbtError(format!("{label} is negative: {raw}")))?;
        if !(minimum..=maximum).contains(&value) {
            return err(format!("{label} {value} is outside {minimum}..={maximum}"));
        }
        Ok(value)
    }

    fn boolean(&mut self, label: &str) -> Result<bool, JrbtError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => err(format!("{label} boolean is {value}; expected 0 or 1")),
        }
    }

    fn activation(&mut self, label: &str) -> Result<u8, JrbtError> {
        let value = self.u8()?;
        if value > 4 {
            return err(format!("{label} activation code {value} is unsupported"));
        }
        Ok(value)
    }

    fn f32vec(&mut self, n: usize, label: &str) -> Result<Vec<f32>, JrbtError> {
        let byte_count = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| JrbtError(format!("{label} byte count overflows")))?;
        let s = self.take(byte_count)?;
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| JrbtError(format!("could not reserve {n} floats for {label}")))?;
        for i in 0..n {
            let o = i * 4;
            let value = f32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
            if !value.is_finite() {
                return err(format!("{label} contains a non-finite value at index {i}"));
            }
            v.push(value);
        }
        Ok(v)
    }
}

/// One named weight tensor read from the JRBT file.
struct Tensor {
    /// Used for retaining the declared dimensions for admission checks.
    shape: Vec<usize>,
    /// Used for holding the flattened row-major float payload.
    data: Vec<f32>,
}

struct Config {
    dim: usize,
    blocks: usize,
    heads: usize,
    dff: usize,
    value_dim: usize,
    n_rel: usize,
    smolgen_hidden_channels: usize,
    smolgen_hidden_size: usize,
    smolgen_gen_size: usize,
    moves_left: bool,
    transposed: bool,
    king_safety: bool,
    use_smolgen: bool,
    act: u8,
    ffn_act: u8,
    smolgen_act: u8,
}

impl Config {
    /// Used for decoding and validating the fixed JRBT v1 architecture header.
    fn read(reader: &mut Reader<'_>) -> Result<Self, JrbtError> {
        let dim = reader.bounded_usize("network width", 1, MAX_NETWORK_WIDTH)?;
        let blocks = reader.bounded_usize("transformer block count", 1, MAX_BLOCKS)?;
        let heads = reader.bounded_usize("attention head count", 1, MAX_HEADS)?;
        let dff = reader.bounded_usize("FFN width", 1, MAX_FFN_WIDTH)?;
        let policy_dim = reader.bounded_usize("policy width", 1, POLICY_SIZE)?;
        let value_dim = reader.bounded_usize("value width", 1, 3)?;
        let n_rel = reader.bounded_usize("relation count", 1, MAX_RELATIONS)?;
        let smolgen_hidden_channels =
            reader.bounded_usize("smolgen hidden channels", 0, MAX_SMOLGEN_WIDTH)?;
        let smolgen_hidden_size =
            reader.bounded_usize("smolgen hidden width", 0, MAX_SMOLGEN_WIDTH)?;
        let smolgen_gen_size =
            reader.bounded_usize("smolgen generated width", 0, MAX_SMOLGEN_WIDTH)?;
        let moves_left = reader.boolean("moves-left")?;
        let transposed = reader.boolean("transposed relations")?;
        let king_safety = reader.boolean("king-safety relations")?;
        let relation_mlp = reader.boolean("relation MLP")?;
        let use_smolgen = reader.boolean("smolgen")?;
        let act = reader.activation("main")?;
        let ffn_act = reader.activation("FFN")?;
        let smolgen_act = reader.activation("smolgen")?;

        if policy_dim != POLICY_SIZE {
            return err(format!("policy_dim {policy_dim} != {POLICY_SIZE}"));
        }
        if value_dim != 3 {
            return err(format!("value_dim {value_dim} != 3"));
        }
        if dim % heads != 0 {
            return err(format!(
                "network width {dim} is not divisible by {heads} attention heads"
            ));
        }
        if relation_mlp {
            return err("relation-MLP JRBT models are unsupported");
        }
        let expected_relations = 8 + usize::from(transposed) * 4 + usize::from(king_safety) * 2;
        if n_rel != expected_relations {
            return err(format!(
                "relation count {n_rel} does not match enabled relation channels {expected_relations}"
            ));
        }
        if use_smolgen
            && (smolgen_hidden_channels == 0 || smolgen_hidden_size == 0 || smolgen_gen_size == 0)
        {
            return err("enabled smolgen dimensions must all be positive");
        }
        checked_product(
            &[TOKENS, smolgen_hidden_channels],
            "smolgen flattened input",
        )?;
        checked_product(&[smolgen_gen_size, heads], "smolgen generated head width")?;

        Ok(Self {
            dim,
            blocks,
            heads,
            dff,
            value_dim,
            n_rel,
            smolgen_hidden_channels,
            smolgen_hidden_size,
            smolgen_gen_size,
            moves_left,
            transposed,
            king_safety,
            use_smolgen,
            act,
            ffn_act,
            smolgen_act,
        })
    }
}

/// Used for multiplying bounded tensor dimensions without wraparound.
fn checked_product(values: &[usize], label: &str) -> Result<usize, JrbtError> {
    let mut product = 1_usize;
    for value in values {
        product = product
            .checked_mul(*value)
            .ok_or_else(|| JrbtError(format!("{label} element count overflows")))?;
        if product > MAX_TENSOR_ELEMENTS {
            return err(format!(
                "{label} has {product} elements; limit is {MAX_TENSOR_ELEMENTS}"
            ));
        }
    }
    Ok(product)
}

/// Used for retrieving one tensor that admission requires by exact name.
fn required_tensor<'a>(
    tensors: &'a HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor, JrbtError> {
    tensors
        .get(name)
        .ok_or_else(|| JrbtError(format!("missing required tensor {name}")))
}

/// Used for requiring one named tensor to have an exact row-major shape.
fn require_shape(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected: &[usize],
) -> Result<(), JrbtError> {
    let tensor = required_tensor(tensors, name)?;
    if tensor.shape != expected {
        return err(format!(
            "tensor {name} has shape {:?}; expected {expected:?}",
            tensor.shape
        ));
    }
    Ok(())
}

/// Used for validating a matrix and returning its bounded output dimension.
fn matrix_output_dimension(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    input: usize,
    maximum: usize,
) -> Result<usize, JrbtError> {
    let tensor = required_tensor(tensors, name)?;
    let [output, actual_input] = tensor.shape.as_slice() else {
        return err(format!(
            "tensor {name} has shape {:?}; expected a two-dimensional matrix",
            tensor.shape
        ));
    };
    if *actual_input != input || *output == 0 || *output > maximum {
        return err(format!(
            "tensor {name} has shape {:?}; expected [1..={maximum}, {input}]",
            tensor.shape
        ));
    }
    Ok(*output)
}

/// Used for proving every unchecked inference slice from admitted shapes.
fn validate_tensors(config: &Config, tensors: &HashMap<String, Tensor>) -> Result<(), JrbtError> {
    let width = config.dim;
    require_shape(tensors, "base.pos_embed", &[1, TOKENS, width])?;
    require_shape(
        tensors,
        "base.square_embed.weight",
        &[width, INPUT_CHANNELS],
    )?;
    require_shape(tensors, "base.square_embed.bias", &[width])?;
    require_shape(tensors, "base.final_norm.weight", &[width])?;
    require_shape(tensors, "base.final_norm.bias", &[width])?;

    let shared_smolgen = tensors.contains_key("base.shared_smolgen_dense.weight");
    if config.use_smolgen && shared_smolgen {
        require_shape(
            tensors,
            "base.shared_smolgen_dense.weight",
            &[4_096, config.smolgen_gen_size],
        )?;
    }
    for block in 0..config.blocks {
        let prefix = format!("base.blocks.{block}.");
        for projection in ["q", "k", "v", "out"] {
            require_shape(
                tensors,
                &format!("{prefix}attn.{projection}.weight"),
                &[width, width],
            )?;
            require_shape(
                tensors,
                &format!("{prefix}attn.{projection}.bias"),
                &[width],
            )?;
        }
        require_shape(tensors, &format!("{prefix}ln1.weight"), &[width])?;
        require_shape(tensors, &format!("{prefix}ln1.bias"), &[width])?;
        require_shape(tensors, &format!("{prefix}ln2.weight"), &[width])?;
        require_shape(tensors, &format!("{prefix}ln2.bias"), &[width])?;
        require_shape(
            tensors,
            &format!("{prefix}ffn1.weight"),
            &[config.dff, width],
        )?;
        require_shape(tensors, &format!("{prefix}ffn1.bias"), &[config.dff])?;
        require_shape(
            tensors,
            &format!("{prefix}ffn2.weight"),
            &[width, config.dff],
        )?;
        require_shape(tensors, &format!("{prefix}ffn2.bias"), &[width])?;
        require_shape(
            tensors,
            &format!("rel_w.{block}"),
            &[config.heads, config.n_rel],
        )?;

        if config.use_smolgen {
            let smolgen = format!("{prefix}attn.smolgen.");
            let flat_input = checked_product(
                &[TOKENS, config.smolgen_hidden_channels],
                "smolgen flattened input",
            )?;
            let generated = checked_product(
                &[config.smolgen_gen_size, config.heads],
                "smolgen generated head width",
            )?;
            require_shape(
                tensors,
                &format!("{smolgen}compress.weight"),
                &[config.smolgen_hidden_channels, width],
            )?;
            require_shape(
                tensors,
                &format!("{smolgen}dense1.weight"),
                &[config.smolgen_hidden_size, flat_input],
            )?;
            require_shape(
                tensors,
                &format!("{smolgen}dense1.bias"),
                &[config.smolgen_hidden_size],
            )?;
            require_shape(
                tensors,
                &format!("{smolgen}ln1.weight"),
                &[config.smolgen_hidden_size],
            )?;
            require_shape(
                tensors,
                &format!("{smolgen}ln1.bias"),
                &[config.smolgen_hidden_size],
            )?;
            require_shape(
                tensors,
                &format!("{smolgen}dense2.weight"),
                &[generated, config.smolgen_hidden_size],
            )?;
            require_shape(tensors, &format!("{smolgen}dense2.bias"), &[generated])?;
            require_shape(tensors, &format!("{smolgen}ln2.weight"), &[generated])?;
            require_shape(tensors, &format!("{smolgen}ln2.bias"), &[generated])?;
            if !shared_smolgen {
                require_shape(
                    tensors,
                    &format!("{smolgen}weight_gen_dense.weight"),
                    &[4_096, config.smolgen_gen_size],
                )?;
            }
        }
    }

    for projection in ["tokens", "q", "k"] {
        require_shape(
            tensors,
            &format!("base.policy_head.{projection}.weight"),
            &[width, width],
        )?;
        require_shape(
            tensors,
            &format!("base.policy_head.{projection}.bias"),
            &[width],
        )?;
    }
    require_shape(
        tensors,
        "base.policy_head.promotion_dense.weight",
        &[4, width],
    )?;

    let value_embed = matrix_output_dimension(
        tensors,
        "base.value_head.embed.weight",
        width,
        MAX_FFN_WIDTH,
    )?;
    require_shape(tensors, "base.value_head.embed.bias", &[value_embed])?;
    let value_wdl = required_tensor(tensors, "base.value_head.wdl.weight")?;
    let [value_outputs, value_hidden] = value_wdl.shape.as_slice() else {
        return err("tensor base.value_head.wdl.weight must be a matrix");
    };
    if *value_outputs != config.value_dim || *value_hidden == 0 || *value_hidden > MAX_FFN_WIDTH {
        return err(format!(
            "tensor base.value_head.wdl.weight has unsupported shape {:?}",
            value_wdl.shape
        ));
    }
    let value_flat = checked_product(&[TOKENS, value_embed], "value-head flattened input")?;
    require_shape(
        tensors,
        "base.value_head.dense1.weight",
        &[*value_hidden, value_flat],
    )?;
    require_shape(tensors, "base.value_head.dense1.bias", &[*value_hidden])?;
    require_shape(tensors, "base.value_head.wdl.bias", &[config.value_dim])?;

    if config.moves_left {
        let moves_embed = matrix_output_dimension(
            tensors,
            "base.moves_left_head.embed.weight",
            width,
            MAX_FFN_WIDTH,
        )?;
        require_shape(tensors, "base.moves_left_head.embed.bias", &[moves_embed])?;
        let moves_out = required_tensor(tensors, "base.moves_left_head.out.weight")?;
        let [moves_outputs, moves_hidden] = moves_out.shape.as_slice() else {
            return err("tensor base.moves_left_head.out.weight must be a matrix");
        };
        if *moves_outputs != 1 || *moves_hidden == 0 || *moves_hidden > MAX_FFN_WIDTH {
            return err(format!(
                "tensor base.moves_left_head.out.weight has unsupported shape {:?}",
                moves_out.shape
            ));
        }
        let moves_flat = checked_product(&[TOKENS, moves_embed], "moves-left flattened input")?;
        require_shape(
            tensors,
            "base.moves_left_head.dense1.weight",
            &[*moves_hidden, moves_flat],
        )?;
        require_shape(
            tensors,
            "base.moves_left_head.dense1.bias",
            &[*moves_hidden],
        )?;
        require_shape(tensors, "base.moves_left_head.out.bias", &[1])?;
    }
    Ok(())
}

// ----------------------------------------------------------------- math ------
#[inline]
fn mish(x: f32) -> f32 {
    // x * tanh(softplus(x)); stable softplus
    let sp = if x > 20.0 { x } else { (x.exp()).ln_1p() };
    x * sp.tanh()
}
#[inline]
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
#[inline]
fn apply_act(x: f32, code: u8) -> f32 {
    match code {
        1 => x.max(0.0),
        2 => mish(x),
        3 => swish(x),
        4 => x.tanh(),
        _ => x,
    }
}

/// Used for reserving and initializing one bounded scratch vector.
///
/// Allocation completes before any element is published, so capacity refusal
/// remains a typed [`JrbtError`] instead of reaching the workspace's aborting
/// allocation handler.
fn filled_vector<T: Clone>(length: usize, value: T, label: &str) -> Result<Vec<T>, JrbtError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| JrbtError(format!("could not reserve {length} elements for {label}")))?;
    output.resize(length, value);
    Ok(output)
}

/// Used for admitting a zero-filled float buffer with checked dimensions.
fn zeroed_floats(dimensions: &[usize], label: &str) -> Result<Vec<f32>, JrbtError> {
    let length = checked_product(dimensions, label)?;
    filled_vector(length, 0.0_f32, label)
}

/// Used for making a fallibly admitted copy of one float scratch buffer.
fn copied_floats(values: &[f32], label: &str) -> Result<Vec<f32>, JrbtError> {
    let mut output = Vec::new();
    output.try_reserve_exact(values.len()).map_err(|_| {
        JrbtError(format!(
            "could not reserve {} floats for {label}",
            values.len()
        ))
    })?;
    output.extend_from_slice(values);
    Ok(output)
}

/// Used for the dense affine map `y[t][o] = sum_i x[t][i]*w[o][i] + b[o]`.
///
/// `x` is row-major `rows x in_dim` and `w` is row-major `out_dim x in_dim`.
/// The result allocation is fallible; successful arithmetic preserves the
/// reference kernel's original loop and accumulation order.
fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_dim: usize,
) -> Result<Vec<f32>, JrbtError> {
    let mut y = zeroed_floats(&[rows, out_dim], "JRBT linear output")?;
    for t in 0..rows {
        let xr = &x[t * in_dim..t * in_dim + in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b.map_or(0.0, |bb| bb[o]);
            for i in 0..in_dim {
                acc += xr[i] * wr[i];
            }
            y[t * out_dim + o] = acc;
        }
    }
    Ok(y)
}

/// row-wise LayerNorm over the last dim (eps 1e-3), scale g, shift b.
fn layernorm(x: &mut [f32], rows: usize, d: usize, g: &[f32], b: &[f32]) {
    for t in 0..rows {
        let r = &mut x[t * d..t * d + d];
        let mut mu = 0.0;
        for &v in r.iter() {
            mu += v;
        }
        mu /= d as f32;
        let mut var = 0.0;
        for &v in r.iter() {
            var += (v - mu) * (v - mu);
        }
        var /= d as f32;
        let inv = 1.0 / (var + 1e-3).sqrt();
        for i in 0..d {
            r[i] = (r[i] - mu) * inv * g[i] + b[i];
        }
    }
}

// ----------------------------------------------------- chess relations -------
struct RelTables {
    knight: [[f32; 64]; 64],
    king: [[f32; 64]; 64],
    pawn: [[f32; 64]; 64],
    rook_ray: Vec<[[i32; 7]; 4]>, // [64][4dir][7step] target square or -1
    bishop_ray: Vec<[[i32; 7]; 4]>,
}

fn inb(r: i32, f: i32) -> bool {
    (0..8).contains(&r) && (0..8).contains(&f)
}

impl RelTables {
    /// Used for constructing the immutable relation geometry fallibly.
    fn try_new() -> Result<Self, JrbtError> {
        let mut knight = [[0.0f32; 64]; 64];
        let mut king = [[0.0f32; 64]; 64];
        let mut pawn = [[0.0f32; 64]; 64];
        let kn = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        let kg = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ];
        for i in 0..64 {
            let (r, f) = ((i / 8) as i32, (i % 8) as i32);
            for &(dr, df) in kn.iter() {
                if inb(r + dr, f + df) {
                    knight[i][((r + dr) * 8 + f + df) as usize] = 1.0;
                }
            }
            for &(dr, df) in kg.iter() {
                if inb(r + dr, f + df) {
                    king[i][((r + dr) * 8 + f + df) as usize] = 1.0;
                }
            }
            for df in [-1, 1] {
                if inb(r + 1, f + df) {
                    pawn[i][((r + 1) * 8 + f + df) as usize] = 1.0;
                }
            }
        }
        let rook_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let bishop_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let ray = |dirs: [(i32, i32); 4]| -> Result<Vec<[[i32; 7]; 4]>, JrbtError> {
            let mut t = filled_vector(64, [[-1_i32; 7]; 4], "JRBT relation rays")?;
            for i in 0..64 {
                let (r, f) = ((i / 8) as i32, (i % 8) as i32);
                for (d, &(dr, df)) in dirs.iter().enumerate() {
                    for s in 0..7 {
                        let (nr, nf) = (r + dr * (s as i32 + 1), f + df * (s as i32 + 1));
                        if inb(nr, nf) {
                            t[i][d][s] = nr * 8 + nf;
                        } else {
                            break;
                        }
                    }
                }
            }
            Ok(t)
        };
        Ok(RelTables {
            knight,
            king,
            pawn,
            rook_ray: ray(rook_dirs)?,
            bishop_ray: ray(bishop_dirs)?,
        })
    }

    /// occlusion-aware slider reachability from every square, given occupancy.
    fn slider_reach(&self, occ: &[f32; 64], rays: &[[[i32; 7]; 4]]) -> Result<Vec<f32>, JrbtError> {
        let mut reach = zeroed_floats(&[64, 64], "JRBT slider reachability")?;
        for i in 0..64 {
            for d in 0..4 {
                let mut blocked = false;
                for s in 0..7 {
                    let t = rays[i][d][s];
                    if t < 0 {
                        break;
                    }
                    let t = t as usize;
                    if !blocked {
                        reach[i * 64 + t] = 1.0;
                    }
                    if occ[t] > 0.5 {
                        blocked = true;
                    }
                }
            }
        }
        Ok(reach)
    }

    /// Used for building `R[n_rel][64][64]` from the 112 channel-major planes.
    ///
    /// Plane `p`, square `sq` lives at `planes[p * 64 + sq]`.
    fn relations(&self, planes: &[f32], cfg: &Config) -> Result<Vec<f32>, JrbtError> {
        let plane = |p: usize, sq: usize| planes[p * 64 + sq];
        let mut our_occ = [0.0f32; 64];
        let mut their_occ = [0.0f32; 64];
        let mut occ = [0.0f32; 64];
        for sq in 0..64 {
            let mut o = 0.0;
            let mut th = 0.0;
            for k in 0..6 {
                o += plane(k, sq);
                th += plane(6 + k, sq);
            }
            our_occ[sq] = o;
            their_occ[sq] = th;
            occ[sq] = (o + th).min(1.0);
        }
        let rook = self.slider_reach(&occ, &self.rook_ray)?;
        let bishop = self.slider_reach(&occ, &self.bishop_ray)?;

        // side attacks: for square i with a piece of type k, its attack set.
        let side_attacks = |base: usize| -> Result<Vec<f32>, JrbtError> {
            let mut a = zeroed_floats(&[64, 64], "JRBT side attacks")?;
            for i in 0..64 {
                let (p, n, b, r, q, k) = (
                    plane(base, i),
                    plane(base + 1, i),
                    plane(base + 2, i),
                    plane(base + 3, i),
                    plane(base + 4, i),
                    plane(base + 5, i),
                );
                if p + n + b + r + q + k < 0.5 {
                    continue;
                }
                for j in 0..64 {
                    let mut v = 0.0;
                    v += p * self.pawn[i][j];
                    v += n * self.knight[i][j];
                    v += b * bishop[i * 64 + j];
                    v += r * rook[i * 64 + j];
                    v += q * (bishop[i * 64 + j] + rook[i * 64 + j]);
                    v += k * self.king[i][j];
                    if v > 0.0 {
                        a[i * 64 + j] = v.min(1.0);
                    }
                }
            }
            Ok(a)
        };
        let our_atk = side_attacks(0)?;
        let their_atk = side_attacks(6)?;

        let mut chans: Vec<Vec<f32>> = Vec::new();
        chans
            .try_reserve_exact(cfg.n_rel)
            .map_err(|_| JrbtError("could not reserve JRBT relation channels".to_owned()))?;
        // helper masks as flat [64*64]
        let flat = |m: &[[f32; 64]; 64]| -> Result<Vec<f32>, JrbtError> {
            let mut v = zeroed_floats(&[64, 64], "JRBT flat relation mask")?;
            for i in 0..64 {
                for j in 0..64 {
                    v[i * 64 + j] = m[i][j];
                }
            }
            Ok(v)
        };
        let mut us_them = zeroed_floats(&[64, 64], "JRBT own attacks on enemies")?;
        let mut them_us = zeroed_floats(&[64, 64], "JRBT enemy attacks on own army")?;
        let mut our_def = zeroed_floats(&[64, 64], "JRBT own defenses")?;
        let mut slider_lines = zeroed_floats(&[64, 64], "JRBT slider lines")?;
        for i in 0..64 {
            for j in 0..64 {
                us_them[i * 64 + j] = our_atk[i * 64 + j] * their_occ[j];
                them_us[i * 64 + j] = their_atk[i * 64 + j] * our_occ[j];
                our_def[i * 64 + j] = our_atk[i * 64 + j] * our_occ[j];
                slider_lines[i * 64 + j] = (rook[i * 64 + j] + bishop[i * 64 + j]).min(1.0);
            }
        }
        chans.push(copied_floats(&our_atk, "JRBT own-attack relation")?);
        chans.push(copied_floats(&their_atk, "JRBT enemy-attack relation")?);
        chans.push(copied_floats(&us_them, "JRBT own-enemy relation")?);
        chans.push(copied_floats(&them_us, "JRBT enemy-own relation")?);
        chans.push(our_def);
        chans.push(flat(&self.knight)?);
        chans.push(flat(&self.king)?);
        chans.push(slider_lines);
        if cfg.transposed {
            let tpose = |m: &[f32]| -> Result<Vec<f32>, JrbtError> {
                let mut t = zeroed_floats(&[64, 64], "JRBT transposed relation")?;
                for i in 0..64 {
                    for j in 0..64 {
                        t[i * 64 + j] = m[j * 64 + i];
                    }
                }
                Ok(t)
            };
            chans.push(tpose(&our_atk)?);
            chans.push(tpose(&their_atk)?);
            chans.push(tpose(&us_them)?);
            chans.push(tpose(&them_us)?);
        }
        if cfg.king_safety {
            // our king zone + their king zone via king adjacency
            let mut our_king = zeroed_floats(&[64], "JRBT own-king plane")?;
            let mut their_king = zeroed_floats(&[64], "JRBT enemy-king plane")?;
            for square in 0..64 {
                our_king[square] = plane(5, square);
                their_king[square] = plane(11, square);
            }
            let zone = |k: &[f32]| -> Result<Vec<f32>, JrbtError> {
                let mut z = zeroed_floats(&[64], "JRBT king zone")?;
                for j in 0..64 {
                    let mut v = k[j];
                    for i in 0..64 {
                        v += k[i] * self.king[i][j];
                    }
                    z[j] = v.min(1.0);
                }
                Ok(z)
            };
            let tz = zone(&their_king)?;
            let oz = zone(&our_king)?;
            let mut c1 = zeroed_floats(&[64, 64], "JRBT own king-zone attacks")?;
            let mut c2 = zeroed_floats(&[64, 64], "JRBT enemy king-zone attacks")?;
            for i in 0..64 {
                for j in 0..64 {
                    c1[i * 64 + j] = our_atk[i * 64 + j] * tz[j];
                    c2[i * 64 + j] = their_atk[i * 64 + j] * oz[j];
                }
            }
            chans.push(c1);
            chans.push(c2);
        }
        // flatten to [n_rel*64*64]
        if chans.len() != cfg.n_rel {
            return err(format!(
                "JRBT relation construction produced {} channels; expected {}",
                chans.len(),
                cfg.n_rel
            ));
        }
        let mut r = zeroed_floats(&[cfg.n_rel, 64, 64], "JRBT relation tensor")?;
        for (c, ch) in chans.iter().enumerate() {
            r[c * 4096..c * 4096 + 4096].copy_from_slice(ch);
        }
        Ok(r)
    }
}

// ------------------------------------------------------------- network -------
/// Loaded JRBT relation-biased transformer and its per-evaluation scratch.
///
/// Owns the parsed configuration, named weight tensors, the 1858-entry LC0
/// policy map, the precomputed relation tables, and the reusable input-plane
/// buffer. Evaluation is MCTS-only, exactly like the BT4 backend.
pub struct JrbtNetwork {
    cfg: Config,
    t: HashMap<String, Tensor>,
    policy_map: Vec<usize>,
    rel: RelTables,
    input_format: Bt4InputFormat,
    plane_buf: Vec<f32>,
}

impl JrbtNetwork {
    /// Used for loading one JRBT network from a local weight file.
    ///
    /// # Arguments
    ///
    /// * `path` - filesystem path of the JRBT weight file
    ///
    /// # Errors
    ///
    /// Returns [`JrbtError`] when the file cannot be read or fails admission.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, JrbtError> {
        let file =
            File::open(path.as_ref()).map_err(|error| JrbtError(format!("read: {error}")))?;
        let declared = file
            .metadata()
            .map_err(|error| JrbtError(format!("metadata: {error}")))?
            .len();
        if declared > MAX_MODEL_BYTES as u64 {
            return err(format!(
                "JRBT file is {declared} bytes; limit is {MAX_MODEL_BYTES}"
            ));
        }
        let capacity = usize::try_from(declared)
            .map_err(|_| JrbtError("JRBT file length does not fit this platform".to_owned()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| JrbtError(format!("could not reserve {capacity} bytes for JRBT file")))?;
        file.take((MAX_MODEL_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| JrbtError(format!("read: {error}")))?;
        if bytes.len() > MAX_MODEL_BYTES {
            return err("JRBT file grew beyond the loader size limit while reading");
        }
        Self::from_bytes(&bytes)
    }

    /// Used for admitting one JRBT network from an in-memory weight image.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete JRBT file image
    ///
    /// # Errors
    ///
    /// Returns [`JrbtError`] when the magic, version, configuration, tensor
    /// table, or policy map is malformed or truncated.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, JrbtError> {
        if bytes.len() > MAX_MODEL_BYTES {
            return err(format!(
                "JRBT buffer is {} bytes; limit is {MAX_MODEL_BYTES}",
                bytes.len()
            ));
        }
        let mut r = Reader { b: bytes, off: 0 };
        if r.take(4)? != MAGIC {
            return err("bad magic (not JRBT)");
        }
        if r.i32()? != VERSION {
            return err("unsupported JRBT version");
        }
        let cfg = Config::read(&mut r)?;
        let n_tensors = r.bounded_usize("tensor count", 1, MAX_TENSORS)?;
        let mut t = HashMap::new();
        t.try_reserve(n_tensors)
            .map_err(|_| JrbtError(format!("could not reserve {n_tensors} JRBT tensors")))?;
        let mut total_elements = 0_usize;
        for tensor_index in 0..n_tensors {
            let name_length = r.bounded_usize("tensor name length", 1, MAX_TENSOR_NAME_BYTES)?;
            let name_bytes = r.take(name_length)?;
            let name_text = std::str::from_utf8(name_bytes)
                .map_err(|_| JrbtError(format!("tensor {tensor_index} name is not valid UTF-8")))?;
            let mut name = String::new();
            name.try_reserve_exact(name_length)
                .map_err(|_| JrbtError(format!("could not reserve tensor {tensor_index} name")))?;
            name.push_str(name_text);
            if t.contains_key(&name) {
                return err(format!("duplicate tensor name {name}"));
            }
            let rank = r.bounded_usize("tensor rank", 1, MAX_TENSOR_RANK)?;
            let mut shape = Vec::new();
            shape
                .try_reserve_exact(rank)
                .map_err(|_| JrbtError(format!("could not reserve shape for tensor {name}")))?;
            for _ in 0..rank {
                shape.push(r.bounded_usize("tensor dimension", 1, MAX_TENSOR_AXIS)?);
            }
            let count = checked_product(&shape, &format!("tensor {name}"))?;
            total_elements = total_elements
                .checked_add(count)
                .ok_or_else(|| JrbtError("total tensor element count overflows".to_owned()))?;
            if total_elements > MAX_TENSOR_ELEMENTS {
                return err(format!(
                    "JRBT tensors contain {total_elements} elements; limit is {MAX_TENSOR_ELEMENTS}"
                ));
            }
            let data = r.f32vec(count, &format!("tensor {name}"))?;
            t.insert(name, Tensor { shape, data });
        }
        let policy_length = r.bounded_usize("policy-map length", 1, POLICY_SIZE)?;
        if policy_length != POLICY_SIZE {
            return err(format!(
                "policy-map length {policy_length} != {POLICY_SIZE}"
            ));
        }
        let mut policy_map = Vec::new();
        policy_map.try_reserve_exact(policy_length).map_err(|_| {
            JrbtError(format!(
                "could not reserve {policy_length} policy-map entries"
            ))
        })?;
        let mut seen_policy = [false; RAW_POLICY_SIZE];
        for output in 0..policy_length {
            let raw = r.i32()?;
            let index = usize::try_from(raw)
                .map_err(|_| JrbtError(format!("policy-map entry {output} is negative: {raw}")))?;
            if index >= RAW_POLICY_SIZE {
                return err(format!(
                    "policy-map entry {output} is {index}; raw-policy limit is {}",
                    RAW_POLICY_SIZE - 1
                ));
            }
            if seen_policy[index] {
                return err(format!(
                    "policy-map raw index {index} is duplicated at output {output}"
                ));
            }
            seen_policy[index] = true;
            policy_map.push(index);
        }
        if r.off != bytes.len() {
            return err(format!(
                "JRBT file has {} trailing bytes",
                bytes.len() - r.off
            ));
        }
        validate_tensors(&cfg, &t)?;
        let rel = RelTables::try_new()?;
        let plane_buf = zeroed_floats(&[INPUT_VALUES], "JRBT input planes")?;
        Ok(JrbtNetwork {
            cfg,
            t,
            policy_map,
            rel,
            input_format: Bt4InputFormat::Classical112,
            plane_buf,
        })
    }

    fn g(&self, name: &str) -> Result<&[f32], JrbtError> {
        self.t
            .get(name)
            .map(|tensor| tensor.data.as_slice())
            .ok_or_else(|| JrbtError(format!("admitted model lost tensor {name}")))
    }

    fn smolgen(&self, blk: usize, tok: &[f32]) -> Result<Vec<f32>, JrbtError> {
        let c = &self.cfg;
        let p = format!("base.blocks.{blk}.attn.smolgen.");
        // compress: Linear(dim, hc, bias=false) over 64 tokens -> [64*hc]
        let h = linear(
            tok,
            TOKENS,
            c.dim,
            self.g(&(p.clone() + "compress.weight"))?,
            None,
            c.smolgen_hidden_channels,
        )?;
        // dense1: [64*hc] -> hidden_size ; but h is [64][hc] row-major = flatten [64*hc]
        let flat_in = TOKENS * c.smolgen_hidden_channels;
        let mut h = linear(
            &h,
            1,
            flat_in,
            self.g(&(p.clone() + "dense1.weight"))?,
            Some(self.g(&(p.clone() + "dense1.bias"))?),
            c.smolgen_hidden_size,
        )?;
        for v in h.iter_mut() {
            *v = apply_act(*v, c.smolgen_act);
        }
        layernorm(
            &mut h,
            1,
            c.smolgen_hidden_size,
            self.g(&(p.clone() + "ln1.weight"))?,
            self.g(&(p.clone() + "ln1.bias"))?,
        );
        let mut h = linear(
            &h,
            1,
            c.smolgen_hidden_size,
            self.g(&(p.clone() + "dense2.weight"))?,
            Some(self.g(&(p.clone() + "dense2.bias"))?),
            c.smolgen_gen_size * c.heads,
        )?;
        for v in h.iter_mut() {
            *v = apply_act(*v, c.smolgen_act);
        }
        layernorm(
            &mut h,
            1,
            c.smolgen_gen_size * c.heads,
            self.g(&(p.clone() + "ln2.weight"))?,
            self.g(&(p.clone() + "ln2.bias"))?,
        );
        // reshape [heads, gen], weight_gen_dense: Linear(gen, 4096) shared
        let wg = if self.t.contains_key("base.shared_smolgen_dense.weight") {
            self.g("base.shared_smolgen_dense.weight")?
        } else {
            self.g(&(p.clone() + "weight_gen_dense.weight"))?
        };
        // out[head] = wg( h[head*gen .. ] ) -> [heads][4096]
        let mut out = zeroed_floats(&[c.heads, 64, 64], "JRBT smolgen output")?;
        for hd in 0..c.heads {
            let hh = &h[hd * c.smolgen_gen_size..hd * c.smolgen_gen_size + c.smolgen_gen_size];
            for o in 0..4096 {
                let wr = &wg[o * c.smolgen_gen_size..o * c.smolgen_gen_size + c.smolgen_gen_size];
                let mut acc = 0.0;
                for i in 0..c.smolgen_gen_size {
                    acc += hh[i] * wr[i];
                }
                out[hd * 4096 + o] = acc;
            }
        }
        Ok(out) // [heads][64*64]
    }

    fn block(&self, blk: usize, x: &mut Vec<f32>, r: &[f32]) -> Result<(), JrbtError> {
        let c = &self.cfg;
        let d = c.dim;
        let hd = d / c.heads;
        let p = format!("base.blocks.{blk}.");
        let q = linear(
            x,
            TOKENS,
            d,
            self.g(&(p.clone() + "attn.q.weight"))?,
            Some(self.g(&(p.clone() + "attn.q.bias"))?),
            d,
        )?;
        let k = linear(
            x,
            TOKENS,
            d,
            self.g(&(p.clone() + "attn.k.weight"))?,
            Some(self.g(&(p.clone() + "attn.k.bias"))?),
            d,
        )?;
        let v = linear(
            x,
            TOKENS,
            d,
            self.g(&(p.clone() + "attn.v.weight"))?,
            Some(self.g(&(p.clone() + "attn.v.bias"))?),
            d,
        )?;
        let smol = if c.use_smolgen {
            Some(self.smolgen(blk, x)?)
        } else {
            None
        };
        let rel_w = self.g(&format!("rel_w.{blk}"))?; // [heads][n_rel]
        let scale = (hd as f32).powf(-0.5);
        // attention output [64][d]
        let mut attn = zeroed_floats(&[TOKENS, d], "JRBT attention output")?;
        let mut logits = zeroed_floats(&[TOKENS], "JRBT attention logits")?;
        for h in 0..c.heads {
            for i in 0..TOKENS {
                // logits over j
                for j in 0..TOKENS {
                    let mut dot = 0.0;
                    for e in 0..hd {
                        dot += q[i * d + h * hd + e] * k[j * d + h * hd + e];
                    }
                    let mut l = dot * scale;
                    if let Some(ref s) = smol {
                        l += s[h * 4096 + i * 64 + j];
                    }
                    let mut rb = 0.0;
                    for rr in 0..c.n_rel {
                        rb += rel_w[h * c.n_rel + rr] * r[rr * 4096 + i * 64 + j];
                    }
                    logits[j] = l + rb;
                }
                // softmax
                let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for j in 0..TOKENS {
                    logits[j] = (logits[j] - mx).exp();
                    sum += logits[j];
                }
                if !sum.is_finite() || sum <= 0.0 {
                    return err(format!(
                        "block {blk} attention produced an invalid softmax sum"
                    ));
                }
                // weighted sum of v -> attn[i, head slice]
                for e in 0..hd {
                    let mut acc = 0.0;
                    for j in 0..TOKENS {
                        acc += logits[j] * v[j * d + h * hd + e];
                    }
                    attn[i * d + h * hd + e] = acc / sum;
                }
            }
        }
        let attn_out = linear(
            &attn,
            TOKENS,
            d,
            self.g(&(p.clone() + "attn.out.weight"))?,
            Some(self.g(&(p.clone() + "attn.out.bias"))?),
            d,
        )?;
        let alpha = (2.0 * c.blocks as f32).powf(-0.25);
        // x = x + attn_out*alpha ; then out = ln1(x) ; ffn ; ln2(out + ffn*alpha)
        for idx in 0..TOKENS * d {
            x[idx] += attn_out[idx] * alpha;
        }
        let mut out = copied_floats(x, "JRBT normalized block state")?;
        layernorm(
            &mut out,
            TOKENS,
            d,
            self.g(&(p.clone() + "ln1.weight"))?,
            self.g(&(p.clone() + "ln1.bias"))?,
        );
        let mut ff = linear(
            &out,
            TOKENS,
            d,
            self.g(&(p.clone() + "ffn1.weight"))?,
            Some(self.g(&(p.clone() + "ffn1.bias"))?),
            c.dff,
        )?;
        for vv in ff.iter_mut() {
            *vv = apply_act(*vv, c.ffn_act);
        }
        let ff = linear(
            &ff,
            TOKENS,
            c.dff,
            self.g(&(p.clone() + "ffn2.weight"))?,
            Some(self.g(&(p.clone() + "ffn2.bias"))?),
            d,
        )?;
        for idx in 0..TOKENS * d {
            out[idx] += ff[idx] * alpha;
        }
        layernorm(
            &mut out,
            TOKENS,
            d,
            self.g(&(p.clone() + "ln2.weight"))?,
            self.g(&(p.clone() + "ln2.bias"))?,
        );
        *x = out;
        Ok(())
    }

    /// Used for running the transformer over one encoded position.
    ///
    /// Returns the 1858-entry LC0 policy logits, the three WDL logits, and
    /// the moves-left head output.
    fn run(&mut self, planes: &[f32]) -> Result<([f32; POLICY_SIZE], [f32; 3], f32), JrbtError> {
        let c = &self.cfg;
        let d = c.dim;
        let r = self.rel.relations(planes, c)?;
        // tokens: tok[t][ch] = planes[ch*64 + t]
        let mut tok = zeroed_floats(&[TOKENS, INPUT_CHANNELS], "JRBT input tokens")?;
        for t in 0..TOKENS {
            for ch in 0..112 {
                tok[t * 112 + ch] = planes[ch * 64 + t];
            }
        }
        let mut h = linear(
            &tok,
            TOKENS,
            112,
            self.g("base.square_embed.weight")?,
            Some(self.g("base.square_embed.bias")?),
            d,
        )?;
        let pos = self.g("base.pos_embed")?;
        for t in 0..TOKENS {
            for ci in 0..d {
                h[t * d + ci] += pos[t * d + ci];
            }
        }
        for blk in 0..c.blocks {
            self.block(blk, &mut h, &r)?;
        }
        layernorm(
            &mut h,
            TOKENS,
            d,
            self.g("base.final_norm.weight")?,
            self.g("base.final_norm.bias")?,
        );
        let policy = self.policy_head(&h)?;
        let wdl = self.value_head(&h)?;
        let moves = if c.moves_left {
            self.moves_head(&h)?
        } else {
            0.0
        };
        if policy.iter().any(|value| !value.is_finite())
            || wdl.iter().any(|value| !value.is_finite())
            || !moves.is_finite()
        {
            return err("JRBT inference produced a non-finite output");
        }
        Ok((policy, wdl, moves))
    }

    fn policy_head(&self, h: &[f32]) -> Result<[f32; POLICY_SIZE], JrbtError> {
        let c = &self.cfg;
        let d = c.dim;
        let dk = (d as f32).sqrt();
        let mut xt = linear(
            h,
            TOKENS,
            d,
            self.g("base.policy_head.tokens.weight")?,
            Some(self.g("base.policy_head.tokens.bias")?),
            d,
        )?;
        for v in xt.iter_mut() {
            *v = apply_act(*v, c.act);
        }
        let q = linear(
            &xt,
            TOKENS,
            d,
            self.g("base.policy_head.q.weight")?,
            Some(self.g("base.policy_head.q.bias")?),
            d,
        )?;
        let k = linear(
            &xt,
            TOKENS,
            d,
            self.g("base.policy_head.k.weight")?,
            Some(self.g("base.policy_head.k.bias")?),
            d,
        )?;
        // qk[i][j] = <q_i,k_j>
        let mut qk = zeroed_floats(&[64, 64], "JRBT policy pair logits")?;
        for i in 0..64 {
            for j in 0..64 {
                let mut acc = 0.0;
                for e in 0..d {
                    acc += q[i * d + e] * k[j * d + e];
                }
                qk[i * 64 + j] = acc;
            }
        }
        // promotion: promotion_dense: Linear(d,4,bias=false) on k[-8:] -> [8,4] ; *dk ; transpose->[4,8]
        let pdw = self.g("base.policy_head.promotion_dense.weight")?; // [4][d]
        let mut po = [[0.0f32; 8]; 4]; // [4][8]
        for (row, s) in (56..64).enumerate() {
            for o in 0..4 {
                let mut acc = 0.0;
                for e in 0..d {
                    acc += k[s * d + e] * pdw[o * d + e];
                }
                po[o][row] = acc * dk;
            }
        }
        // po = po[:3] + po[3:4]
        let mut poff = [[0.0f32; 8]; 3];
        for o in 0..3 {
            for j in 0..8 {
                poff[o][j] = po[o][j] + po[3][j];
            }
        }
        // n_promo_logits = qk[-16:-8, -8:] = from 48..56, to 56..64 -> [8][8]
        // promo[8from][8to][3piece] = npl + poff[piece][to]
        // raw = concat( (qk/dk).flatten()[4096], (promo.reshape(8,24)/dk).flatten()[192] )
        let mut raw = zeroed_floats(&[RAW_POLICY_SIZE], "JRBT raw policy logits")?;
        for i in 0..64 {
            for j in 0..64 {
                raw[i * 64 + j] = qk[i * 64 + j] / dk;
            }
        }
        // promo block: for fr in 0..8 (square 48+fr), to in 0..8, piece in 0..3
        for fr in 0..8 {
            let npl_row = 48 + fr;
            for to in 0..8 {
                let npl = qk[npl_row * 64 + (56 + to)];
                for piece in 0..3 {
                    let val = (npl + poff[piece][to]) / dk;
                    // reshape(8,24): index = fr*24 + to*3 + piece ; flattened after the 4096
                    raw[4096 + fr * 24 + to * 3 + piece] = val;
                }
            }
        }
        let mut out = [0.0f32; POLICY_SIZE];
        for (o, &idx) in self.policy_map.iter().enumerate() {
            out[o] = raw[idx];
        }
        Ok(out)
    }

    fn value_head(&self, h: &[f32]) -> Result<[f32; 3], JrbtError> {
        let c = &self.cfg;
        // embed: Linear(d,128) per token -> mish -> flatten [64*128] -> dense1(->128) mish -> wdl(->3)
        let emb_ch = self.g("base.value_head.embed.weight")?.len() / c.dim; // out channels
        let mut e = linear(
            h,
            TOKENS,
            c.dim,
            self.g("base.value_head.embed.weight")?,
            Some(self.g("base.value_head.embed.bias")?),
            emb_ch,
        )?;
        for v in e.iter_mut() {
            *v = apply_act(*v, c.act);
        }
        let flat_in = TOKENS * emb_ch;
        let hid = self.g("base.value_head.wdl.weight")?.len() / c.value_dim; // dense1 out
        let mut d1 = linear(
            &e,
            1,
            flat_in,
            self.g("base.value_head.dense1.weight")?,
            Some(self.g("base.value_head.dense1.bias")?),
            hid,
        )?;
        for v in d1.iter_mut() {
            *v = apply_act(*v, c.act);
        }
        let wdl = linear(
            &d1,
            1,
            hid,
            self.g("base.value_head.wdl.weight")?,
            Some(self.g("base.value_head.wdl.bias")?),
            c.value_dim,
        )?;
        // softmax
        let mx = wdl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut ex = [0.0f32; 3];
        let mut sum = 0.0;
        for i in 0..3 {
            ex[i] = (wdl[i] - mx).exp();
            sum += ex[i];
        }
        if !sum.is_finite() || sum <= 0.0 {
            return err("JRBT value head produced an invalid softmax sum");
        }
        Ok([ex[0] / sum, ex[1] / sum, ex[2] / sum])
    }

    fn moves_head(&self, h: &[f32]) -> Result<f32, JrbtError> {
        let c = &self.cfg;
        let emb_ch = self.g("base.moves_left_head.embed.weight")?.len() / c.dim;
        let mut e = linear(
            h,
            TOKENS,
            c.dim,
            self.g("base.moves_left_head.embed.weight")?,
            Some(self.g("base.moves_left_head.embed.bias")?),
            emb_ch,
        )?;
        for v in e.iter_mut() {
            *v = apply_act(*v, c.act);
        }
        let flat_in = TOKENS * emb_ch;
        let hid = self.g("base.moves_left_head.out.weight")?.len();
        let mut d1 = linear(
            &e,
            1,
            flat_in,
            self.g("base.moves_left_head.dense1.weight")?,
            Some(self.g("base.moves_left_head.dense1.bias")?),
            hid,
        )?;
        for v in d1.iter_mut() {
            *v = apply_act(*v, c.act);
        }
        let o = linear(
            &d1,
            1,
            hid,
            self.g("base.moves_left_head.out.weight")?,
            Some(self.g("base.moves_left_head.out.bias")?),
            1,
        )?;
        Ok(o[0].max(0.0))
    }
}

impl Evaluator for JrbtNetwork {
    fn evaluate(&mut self, position: &Position) -> i32 {
        let pv = self.evaluate_policy_value(position, &[]);
        // value in [-1,1] -> centipawns (LC0-style logistic inverse, clamped)
        let v = pv.value.clamp(-0.999, 0.999);
        (111.7 * (1.5620688 * v).tan()) as i32
    }

    fn evaluate_policy_value(&mut self, position: &Position, legal_moves: &[Move]) -> PolicyValue {
        let mut planes = std::mem::take(&mut self.plane_buf);
        let Ok(transform) = encode_lc0_fen_position_into(position, self.input_format, &mut planes)
        else {
            self.plane_buf = planes;
            return PolicyValue::from_centipawns(0);
        };
        let prediction = self.run(&planes);
        self.plane_buf = planes;
        let Ok((policy, wdl, _moves)) = prediction else {
            return PolicyValue::from_centipawns(0);
        };
        let value = wdl[0] - wdl[2];
        let draw = wdl[1];
        if legal_moves.is_empty() {
            return PolicyValue::with_logits(value, draw, Vec::new());
        }
        match gather_legal_policy_logits(position, legal_moves, &policy, transform) {
            Ok(logits) => PolicyValue::with_logits(value, draw, logits),
            Err(_) => PolicyValue::with_logits(value, draw, Vec::new()),
        }
    }

    fn allows_mcts_quiescence(&self) -> bool {
        false
    }
}
