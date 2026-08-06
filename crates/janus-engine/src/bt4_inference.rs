#![allow(
    // Fixed architecture dimensions fit every destination type. Transcendental
    // results intentionally narrow at the Java reference's f64-to-f32 boundary.
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

//! Cache-conscious scalar inference for the pinned full CRTK BT4 v2 network.
//!
//! Activations use feature-major `[feature][token]` storage. A dense output
//! worker consequently reads each row-major weight once and applies it to the
//! contiguous 64-token lane, instead of re-reading the 706 MiB parameter set
//! once per token. Scoped workers own disjoint output-feature ranges; no
//! synchronization, `unsafe`, external BLAS, or nondeterministic reduction is
//! involved. Every scalar output retains input-feature accumulation order.

use crate::bt4::{
    Bt4Activation, Bt4Architecture, Bt4Error, Bt4ErrorKind, Bt4InputEmbedding, Bt4InputFormat,
    Bt4Model, Bt4TensorInfo,
};
use crate::bt4_encoding::{
    encode_history_into, encode_lc0_fen_position_into, gather_legal_internal_logits,
    Bt4EncodingError, HISTORY,
};
use crate::bt4_gpu::{Bt4Backend, Bt4BackendStatus};
#[cfg(feature = "gpu")]
use crate::bt4_gpu::{
    Bt4GpuClient, Bt4GpuError, INPUT_FLOATS as GPU_INPUT_FLOATS, POLICY_FLOATS as GPU_POLICY_FLOATS,
};
use crate::evaluator::{Evaluator, PolicyValue};
use crate::threading::spawn_scoped_or_run;
use janus_core::{Move, Position};
use std::path::Path;

/// Used for sizing the channel dimension expected by the pinned classical
/// input encoder.
pub const INPUT_CHANNELS: usize = 112;
/// Used for sizing the board-square token dimension of the full BT4 network.
pub const TOKENS: usize = 64;
/// Used for sizing the compressed LC0 attention-policy output.
pub const POLICY_SIZE: usize = 1_858;
/// Used for sizing the internal from/to plus promotion-offset policy.
pub const INTERNAL_POLICY_SIZE: usize = 67 * TOKENS;
/// Used for capping the scoped CPU team accepted by the scalar dense kernels.
pub const MAX_INFERENCE_THREADS: usize = 16;

/// Used for sizing the fixed model width of the pinned network.
const EMBEDDING: usize = 1_024;
/// Used for counting the transformer body blocks in the pinned network.
const ENCODER_LAYERS: usize = 15;
/// Used for counting the attention heads per body block.
const ATTENTION_HEADS: usize = 32;
/// Used for sizing the feed-forward hidden width of every body and input
/// FFN.
const FFN_HIDDEN: usize = 1_536;
/// Used for sizing the per-token smolgen compression width.
const SMOLGEN_CHANNELS: usize = 32;
/// Used for sizing the first smolgen vector-projection width.
const SMOLGEN_HIDDEN: usize = 256;
/// Used for counting the smolgen coefficients generated for each attention
/// head.
const SMOLGEN_PER_HEAD: usize = 256;
/// Used for counting the attention-bias cells generated for one head.
const ATTENTION_MAP: usize = TOKENS * TOKENS;
/// Used for counting the leading input features consumed by `PE_DENSE`
/// preprocessing.
const PREPROC_CHANNELS: usize = 12;
/// Used for skipping scoped thread setup below this dense operation size.
const PARALLEL_DENSE_WORK: usize = 4_000_000;
/// Used for sizing the win/draw/loss output.
const WDL_OUTPUTS: usize = 3;
/// Used for sizing the from-to portion of the internal attention-policy
/// tensor.
const FROM_TO_POLICY_SIZE: usize = TOKENS * TOKENS;
/// Used for accepting knight offsets in the geometry-derived policy gather.
const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-1, 2),
    (-2, 1),
];

/// Used for checking whether an architecture is the exact full v2 shape
/// implemented by this scalar backend.
///
/// # Arguments
///
/// * `architecture` - decoded container architecture header
///
/// # Returns
///
/// `true` only when every dimension, activation, and v2 extension flag
/// matches the pinned network.
#[must_use]
pub(crate) const fn architecture_supported(architecture: &Bt4Architecture) -> bool {
    if !matches!(architecture.input_format, Bt4InputFormat::Classical112)
        || !matches!(
            architecture.input_embedding,
            Bt4InputEmbedding::PositionDense
        )
        || architecture.input_channels != INPUT_CHANNELS as u32
        || architecture.tokens != TOKENS as u32
        || architecture.embedding_size != EMBEDDING as u32
        || architecture.encoder_layers != ENCODER_LAYERS as u32
        || architecture.attention_heads != ATTENTION_HEADS as u32
        || architecture.policy_size != POLICY_SIZE as u32
    {
        return false;
    }
    let Some(extension) = architecture.v2.as_ref() else {
        return false;
    };
    extension.ffn_hidden_size == FFN_HIDDEN as u32
        && extension.smolgen_hidden_channels == SMOLGEN_CHANNELS as u32
        && extension.smolgen_hidden_size == SMOLGEN_HIDDEN as u32
        && extension.smolgen_per_head_dim == SMOLGEN_PER_HEAD as u32
        && extension.smolgen_global_size == ATTENTION_MAP as u32
        && matches!(extension.default_activation, Bt4Activation::Mish)
        && matches!(extension.smolgen_activation, Bt4Activation::Swish)
        && matches!(extension.ffn_activation, Bt4Activation::Mish)
        && extension.has_input_preproc
        && extension.has_input_embedding_ffn
        && extension.has_input_gates
        && extension.has_smolgen
}

/// Shape and size metadata for a validated executable model.
///
/// Every dimension mirrors the pinned full BT4 v2 network confirmed by
/// `architecture_supported`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bt4Info {
    /// Used for recording the board-plane interpretation required before
    /// inference.
    pub input_format: Bt4InputFormat,
    /// Used for recording the number of caller-provided channel-major input
    /// features.
    pub input_channels: usize,
    /// Used for recording the number of square tokens processed together.
    pub tokens: usize,
    /// Used for recording the main transformer feature width.
    pub embedding_size: usize,
    /// Used for recording the number of body transformer blocks.
    pub encoder_layers: usize,
    /// Used for recording the number of attention heads in every body block.
    pub attention_heads: usize,
    /// Used for recording the compressed policy-logit count.
    pub policy_size: usize,
    /// Used for recording the total finite parameters retained by the
    /// network.
    pub parameter_count: u64,
}

/// Complete policy and WDL prediction from the encoded side-to-move view.
///
/// The policy is already gathered into compressed LC0 order; the value is
/// derived from the WDL head.
#[derive(Clone, Debug, PartialEq)]
pub struct Bt4Prediction {
    /// Used for holding the geometry-compressed attention-policy logits.
    pub policy: Vec<f32>,
    /// Used for holding the probabilities ordered win, draw, loss.
    pub wdl: [f32; WDL_OUTPUTS],
    /// Used for holding the expected game result, computed as win minus
    /// loss.
    pub value: f32,
}

/// Immutable row-major affine layer.
///
/// The forward pass reads each `[output][input]` weight once and applies it
/// to a contiguous token lane of feature-major activations.
#[derive(Debug)]
struct Dense {
    /// Used for recording the input feature count.
    input: usize,
    /// Used for recording the output feature count and bias width.
    output: usize,
    /// Used for holding the coefficients in `[output][input]` order.
    weights: Vec<f32>,
    /// Used for holding one additive value per output feature.
    bias: Vec<f32>,
}

impl Dense {
    /// Used for computing feature-major token lanes with deterministic
    /// input-feature accumulation and optional disjoint output-feature
    /// workers.
    ///
    /// Work below [`PARALLEL_DENSE_WORK`] stays on the calling thread; larger
    /// layers split output features across scoped workers while every scalar
    /// output keeps single-worker accumulation order.
    ///
    /// # Arguments
    ///
    /// * `input` - feature-major `[input][tokens]` activations
    /// * `tokens` - token-lane width shared by input and output
    /// * `output` - feature-major `[output][tokens]` destination
    /// * `threads` - configured maximum worker count
    fn forward(&self, input: &[f32], tokens: usize, output: &mut [f32], threads: usize) {
        debug_assert_eq!(input.len(), self.input * tokens);
        debug_assert_eq!(output.len(), self.output * tokens);
        let work = self
            .input
            .saturating_mul(self.output)
            .saturating_mul(tokens);
        let workers = if work >= PARALLEL_DENSE_WORK {
            threads.min(self.output)
        } else {
            1
        };
        if workers <= 1 {
            self.forward_features(input, tokens, output, 0);
            return;
        }
        let features_per_worker = self.output.div_ceil(workers);
        let values_per_worker = features_per_worker * tokens;
        std::thread::scope(|scope| {
            let mut chunks = output.chunks_mut(values_per_worker).enumerate();
            let (_, caller_output) = chunks.next().expect("dense output is nonempty");
            for (chunk_index, output_chunk) in chunks {
                let first_feature = chunk_index * features_per_worker;
                spawn_scoped_or_run(scope, "janus-bt4-dense", move || {
                    self.forward_features(input, tokens, output_chunk, first_feature);
                });
            }
            self.forward_features(input, tokens, caller_output, 0);
        });
    }

    /// Used for computing a contiguous range of output features owned by one
    /// worker.
    ///
    /// # Arguments
    ///
    /// * `input` - feature-major `[input][tokens]` activations
    /// * `tokens` - token-lane width
    /// * `output` - destination slice covering whole output-feature lanes
    /// * `first_feature` - global index of the first output feature written
    fn forward_features(
        &self,
        input: &[f32],
        tokens: usize,
        output: &mut [f32],
        first_feature: usize,
    ) {
        debug_assert_eq!(output.len() % tokens, 0);
        for (local_output, output_lane) in output.chunks_exact_mut(tokens).enumerate() {
            let output_feature = first_feature + local_output;
            output_lane.fill(self.bias[output_feature]);
            let weight_base = output_feature * self.input;
            for input_feature in 0..self.input {
                let weight = self.weights[weight_base + input_feature];
                let input_base = input_feature * tokens;
                for token in 0..tokens {
                    output_lane[token] += weight * input[input_base + token];
                }
            }
        }
    }
}

/// Input projection, gates, and embedding FFN.
///
/// These layers turn the encoded classical planes into the feature-major
/// trunk activation consumed by the transformer body.
#[derive(Debug)]
struct InputStack {
    /// Used for holding the global `PE_DENSE` preprocessor.
    preproc: Dense,
    /// Used for holding the main per-token feature projection.
    embedding: Dense,
    /// Used for holding the scale vector for embedding layer normalization.
    embedding_ln_gamma: Vec<f32>,
    /// Used for holding the bias vector for embedding layer normalization.
    embedding_ln_beta: Vec<f32>,
    /// Used for holding the multiplicative `[feature][token]` input gate.
    mult_gate: Vec<f32>,
    /// Used for holding the additive `[feature][token]` input gate.
    add_gate: Vec<f32>,
    /// Used for holding the first embedding-FFN projection.
    ffn_in: Dense,
    /// Used for holding the second embedding-FFN projection.
    ffn_out: Dense,
    /// Used for holding the scale vector for the embedding-FFN
    /// post-normalization.
    ffn_ln_gamma: Vec<f32>,
    /// Used for holding the bias vector for the embedding-FFN
    /// post-normalization.
    ffn_ln_beta: Vec<f32>,
}

/// Per-block smolgen attention-bias generator.
///
/// Smolgen compresses the board activation, projects it through two dense
/// layers, and produces per-head coefficients later expanded through the
/// globally shared projection matrix.
#[derive(Debug)]
struct Smolgen {
    /// Used for holding the per-token compression from the body width.
    compress: Dense,
    /// Used for holding the projection of the flattened compressed board.
    dense1: Dense,
    /// Used for holding the first vector-normalization scale.
    ln1_gamma: Vec<f32>,
    /// Used for holding the first vector-normalization bias.
    ln1_beta: Vec<f32>,
    /// Used for holding the projection to all per-head coefficients.
    dense2: Dense,
    /// Used for holding the second vector-normalization scale.
    ln2_gamma: Vec<f32>,
    /// Used for holding the second vector-normalization bias.
    ln2_beta: Vec<f32>,
}

/// Multi-head self-attention parameters.
#[derive(Debug)]
struct Attention {
    /// Used for holding the query projection.
    query: Dense,
    /// Used for holding the key projection.
    key: Dense,
    /// Used for holding the value projection.
    value: Dense,
    /// Used for holding the projection after concatenating the attention
    /// heads.
    output: Dense,
    /// Used for holding the data-dependent attention-bias generator.
    smolgen: Smolgen,
}

/// One post-layer-normalized transformer body block.
#[derive(Debug)]
struct EncoderBlock {
    /// Used for holding the self-attention sublayer.
    attention: Attention,
    /// Used for holding the first feed-forward projection.
    ffn_in: Dense,
    /// Used for holding the second feed-forward projection.
    ffn_out: Dense,
    /// Used for holding the attention residual-normalization scale.
    ln1_gamma: Vec<f32>,
    /// Used for holding the attention residual-normalization bias.
    ln1_beta: Vec<f32>,
    /// Used for holding the FFN residual-normalization scale.
    ln2_gamma: Vec<f32>,
    /// Used for holding the FFN residual-normalization bias.
    ln2_beta: Vec<f32>,
}

/// Attention policy-head tensors.
#[derive(Debug)]
struct PolicyHead {
    /// Used for holding the per-token projection from the body.
    embedding: Dense,
    /// Used for holding the query projection for from squares.
    query: Dense,
    /// Used for holding the key projection for destination squares.
    key: Dense,
    /// Used for holding three promoted-piece offsets plus the shared knight
    /// baseline projection.
    promotion_weights: Vec<f32>,
}

/// Dense WDL value-head tensors.
#[derive(Debug)]
struct ValueHead {
    /// Used for holding the per-token projection from the body.
    embedding: Dense,
    /// Used for holding the hidden projection after token-major flattening.
    fc1: Dense,
    /// Used for holding the three-logit WDL projection.
    fc2: Dense,
}

/// Complete immutable executable parameter set.
///
/// All tensors are typed and shape-checked at load time; nothing here changes
/// during prediction.
#[derive(Debug)]
struct Weights {
    /// Used for holding the public model metadata.
    info: Bt4Info,
    /// Used for holding the input projection stack.
    input: InputStack,
    /// Used for holding the ordered transformer body blocks.
    encoders: Vec<EncoderBlock>,
    /// Used for holding the shared smolgen matrix in
    /// `[attention_cell][per_head_feature]` order.
    smolgen_projection: Vec<f32>,
    /// Used for holding the attention policy head.
    policy: PolicyHead,
    /// Used for holding the WDL value head.
    value: ValueHead,
    /// Used for holding the geometry-derived internal indices in
    /// compressed-policy order.
    policy_gather: Vec<usize>,
    /// Used for holding the positive layer-normalization epsilon from the
    /// container header.
    layer_norm_epsilon: f32,
    /// Used for holding the `DeepNet` residual scale applied to the
    /// input-embedding FFN and every body sublayer output.
    residual_alpha: f32,
}

/// Reusable model-sized activation buffers.
///
/// Every buffer is allocated once from validated layer dimensions and
/// overwritten on each prediction, so no per-inference allocation occurs.
#[derive(Debug)]
struct Workspace {
    /// Used for holding the caller-provided canonical `[channel][token]`
    /// planes.
    encoded: Vec<f32>,
    /// Used for holding the token-major leading-feature vector for `PE_DENSE`.
    preproc_input: Vec<f32>,
    /// Used for holding the token-major `PE_DENSE` projection output.
    preproc_output: Vec<f32>,
    /// Used for holding the feature-major concatenated base and preprocessed
    /// input.
    embedding_input: Vec<f32>,
    /// Used for holding the current body activation.
    flow: Vec<f32>,
    /// Used for holding the attention or residual output activation.
    next: Vec<f32>,
    /// Used for holding the query projection scratch.
    query: Vec<f32>,
    /// Used for holding the key projection scratch.
    key: Vec<f32>,
    /// Used for holding the value projection scratch.
    value: Vec<f32>,
    /// Used for holding the concatenated attention-head output.
    combined: Vec<f32>,
    /// Used for holding the feed-forward hidden activation.
    ffn_hidden: Vec<f32>,
    /// Used for holding the feature-major smolgen compression output.
    smolgen_compressed: Vec<f32>,
    /// Used for holding the token-major flattened smolgen compression input.
    smolgen_flat: Vec<f32>,
    /// Used for holding the first smolgen vector projection.
    smolgen_mid: Vec<f32>,
    /// Used for holding the per-head smolgen coefficients.
    smolgen_generated: Vec<f32>,
    /// Used for holding the per-head square-to-square attention biases.
    smolgen_bias: Vec<f32>,
    /// Used for holding the policy embedding activation.
    policy_flow: Vec<f32>,
    /// Used for holding the policy query projection.
    policy_query: Vec<f32>,
    /// Used for holding the policy key projection.
    policy_key: Vec<f32>,
    /// Used for holding the internal 67-plane attention policy.
    internal_policy: Vec<f32>,
    /// Used for holding the value embedding activation.
    value_embedding: Vec<f32>,
    /// Used for holding the token-major flattened value embedding.
    value_flat: Vec<f32>,
    /// Used for holding the value hidden activation.
    value_hidden: Vec<f32>,
    /// Used for holding the WDL logits before softmax.
    value_logits: Vec<f32>,
}

impl Workspace {
    /// Used for allocating every scratch tensor once from validated layer
    /// dimensions.
    ///
    /// # Arguments
    ///
    /// * `weights` - typed layers whose shapes size the buffers
    ///
    /// # Returns
    ///
    /// Zero-filled workspace covering the body, both heads, and smolgen.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when any buffer reservation fails.
    fn new(weights: &Weights) -> Result<Self, Bt4Error> {
        let trunk = EMBEDDING * TOKENS;
        let preproc_output = weights.input.preproc.output;
        let embedding_input = weights.input.embedding.input * TOKENS;
        let policy_width = weights.policy.embedding.output * TOKENS;
        let policy_projection = weights.policy.query.output * TOKENS;
        let value_width = weights.value.embedding.output * TOKENS;
        Ok(Self {
            encoded: zeroed(INPUT_CHANNELS * TOKENS, "BT4 encoded input")?,
            preproc_input: zeroed(PREPROC_CHANNELS * TOKENS, "BT4 preproc input")?,
            preproc_output: zeroed(preproc_output, "BT4 preproc output")?,
            embedding_input: zeroed(embedding_input, "BT4 embedding input")?,
            flow: zeroed(trunk, "BT4 body flow")?,
            next: zeroed(trunk, "BT4 body next")?,
            query: zeroed(trunk, "BT4 attention query")?,
            key: zeroed(trunk, "BT4 attention key")?,
            value: zeroed(trunk, "BT4 attention value")?,
            combined: zeroed(trunk, "BT4 combined attention")?,
            ffn_hidden: zeroed(FFN_HIDDEN * TOKENS, "BT4 FFN hidden")?,
            smolgen_compressed: zeroed(SMOLGEN_CHANNELS * TOKENS, "BT4 smolgen compression")?,
            smolgen_flat: zeroed(SMOLGEN_CHANNELS * TOKENS, "BT4 smolgen flat")?,
            smolgen_mid: zeroed(SMOLGEN_HIDDEN, "BT4 smolgen hidden")?,
            smolgen_generated: zeroed(ATTENTION_HEADS * SMOLGEN_PER_HEAD, "BT4 smolgen generated")?,
            smolgen_bias: zeroed(ATTENTION_HEADS * ATTENTION_MAP, "BT4 smolgen bias")?,
            policy_flow: zeroed(policy_width, "BT4 policy embedding")?,
            policy_query: zeroed(policy_projection, "BT4 policy query")?,
            policy_key: zeroed(policy_projection, "BT4 policy key")?,
            internal_policy: zeroed(INTERNAL_POLICY_SIZE, "BT4 internal policy")?,
            value_embedding: zeroed(value_width, "BT4 value embedding")?,
            value_flat: zeroed(value_width, "BT4 flattened value")?,
            value_hidden: zeroed(weights.value.fc1.output, "BT4 value hidden")?,
            value_logits: zeroed(WDL_OUTPUTS, "BT4 WDL logits")?,
        })
    }

    /// Used for clearing every reusable activation before and after backend
    /// admission probing.
    #[cfg(feature = "gpu")]
    fn clear(&mut self) {
        self.encoded.fill(0.0);
        self.preproc_input.fill(0.0);
        self.preproc_output.fill(0.0);
        self.embedding_input.fill(0.0);
        self.flow.fill(0.0);
        self.next.fill(0.0);
        self.query.fill(0.0);
        self.key.fill(0.0);
        self.value.fill(0.0);
        self.combined.fill(0.0);
        self.ffn_hidden.fill(0.0);
        self.smolgen_compressed.fill(0.0);
        self.smolgen_flat.fill(0.0);
        self.smolgen_mid.fill(0.0);
        self.smolgen_generated.fill(0.0);
        self.smolgen_bias.fill(0.0);
        self.policy_flow.fill(0.0);
        self.policy_query.fill(0.0);
        self.policy_key.fill(0.0);
        self.internal_policy.fill(0.0);
        self.value_embedding.fill(0.0);
        self.value_flat.fill(0.0);
        self.value_hidden.fill(0.0);
        self.value_logits.fill(0.0);
    }
}

/// Cursor that moves already-decoded tensors into typed layers in manifest
/// order while checking every expected hierarchical name.
struct TensorCursor<'a> {
    /// Used for holding the structural manifest produced before any large
    /// allocation.
    manifest: &'a [Bt4TensorInfo],
    /// Used for holding the owned decoded payloads moved directly into
    /// immutable layers.
    payloads: std::vec::IntoIter<Vec<f32>>,
    /// Used for tracking the index of the next tensor in both sequences.
    index: usize,
}

impl<'a> TensorCursor<'a> {
    /// Used for creating a cursor over equal-length manifest and payload
    /// sequences.
    ///
    /// # Arguments
    ///
    /// * `manifest` - structural tensor descriptions in container order
    /// * `payloads` - decoded tensor values in the same order
    ///
    /// # Returns
    ///
    /// Cursor positioned at the first tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when the sequences have different lengths.
    fn new(manifest: &'a [Bt4TensorInfo], payloads: Vec<Vec<f32>>) -> Result<Self, Bt4Error> {
        if manifest.len() != payloads.len() {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                "BT4 decoded tensor table does not match its manifest",
            ));
        }
        Ok(Self {
            manifest,
            payloads: payloads.into_iter(),
            index: 0,
        })
    }

    /// Used for moving one exact-name, exact-length tensor out of the
    /// decoded sequence.
    ///
    /// # Arguments
    ///
    /// * `name` - expected hierarchical tensor name
    /// * `elements` - exact element count the executable shape requires
    ///
    /// # Returns
    ///
    /// Owned tensor values, advancing the cursor by one.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when the manifest is exhausted, the name or
    /// element count mismatches, or the decoded payload is absent or changed
    /// length after validation.
    fn tensor(&mut self, name: &str, elements: usize) -> Result<Vec<f32>, Bt4Error> {
        let info = self.manifest.get(self.index).ok_or_else(|| {
            Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("missing executable BT4 tensor {name}"),
            )
        })?;
        if info.name != name {
            return Err(Bt4Error::new(
                Bt4ErrorKind::UnsupportedShape,
                format!(
                    "executable BT4 tensor {} is {}; expected {name}",
                    self.index, info.name
                ),
            ));
        }
        if info.elements != elements as u64 {
            return Err(Bt4Error::new(
                Bt4ErrorKind::UnsupportedShape,
                format!(
                    "{name} has {} values; executable shape requires {elements}",
                    info.elements
                ),
            ));
        }
        let values = self.payloads.next().ok_or_else(|| {
            Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("decoded payload for {name} is absent"),
            )
        })?;
        if values.len() != elements {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("decoded {name} length changed after validation"),
            ));
        }
        self.index += 1;
        Ok(values)
    }

    /// Used for moving a row-major dense layer with an exact executable
    /// shape.
    ///
    /// # Arguments
    ///
    /// * `name` - layer name prefix; `.weights` and `.bias` are appended
    /// * `input` - input feature count
    /// * `output` - output feature count and bias width
    ///
    /// # Returns
    ///
    /// Typed dense layer built from the next two tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when either underlying tensor fails validation.
    fn dense(&mut self, name: &str, input: usize, output: usize) -> Result<Dense, Bt4Error> {
        let weights = self.tensor(&format!("{name}.weights"), input * output)?;
        let bias = self.tensor(&format!("{name}.bias"), output)?;
        Ok(Dense {
            input,
            output,
            weights,
            bias,
        })
    }

    /// Used for requiring exact exhaustion after the final value-head
    /// tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when the loader did not consume the exact tensor
    /// manifest.
    fn finish(mut self) -> Result<(), Bt4Error> {
        if self.index != self.manifest.len() || self.payloads.next().is_some() {
            return Err(Bt4Error::new(
                Bt4ErrorKind::UnsupportedShape,
                "executable BT4 loader did not consume the exact tensor manifest",
            ));
        }
        Ok(())
    }
}

/// Loaded scalar BT4 policy/value network with reusable inference buffers.
///
/// The model is intentionally mutable at prediction time because all large
/// activations are retained and overwritten. Create one network per concurrent
/// search; its internal dense workers remain bounded by
/// [`MAX_INFERENCE_THREADS`].
#[derive(Debug)]
pub struct Bt4Network {
    /// Used for holding the immutable finite parameters.
    weights: Weights,
    /// Used for holding the reusable activations.
    workspace: Workspace,
    /// Used for selecting scoped output-feature workers for large dense
    /// layers.
    inference_threads: usize,
    /// Used for retaining at most eight real game positions ending at the
    /// configured search root.
    root_history: Vec<Position>,
    /// Used for retaining the root history plus positions on the current
    /// MCTS descent.
    active_history: Vec<Position>,
    /// Used for holding the persistent fault-isolated `OpenCL` helper when GPU
    /// execution was admitted.
    #[cfg(feature = "gpu")]
    gpu: Option<Bt4GpuClient>,
    /// Used for reporting the requested and active execution backend to UCI
    /// callers.
    backend_status: Bt4BackendStatus,
}

impl Bt4Network {
    /// Used for opening, structurally validating, bulk-decoding, and typing
    /// the pinned full BT4 v2 model.
    ///
    /// # Arguments
    ///
    /// * `path` - model container path
    ///
    /// # Returns
    ///
    /// CPU-backed network ready for prediction.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] for I/O or format failures, non-finite parameters,
    /// resource-limit violations, unsupported BT4 variants, or scratch
    /// allocation failure.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Bt4Error> {
        Self::load_with_backend(path, Bt4Backend::Cpu, 0)
    }

    /// Used for opening the pinned model and selecting CPU or persistent
    /// `OpenCL` execution.
    ///
    /// `Auto` attempts the helper and records an observable CPU fallback reason
    /// if process startup, model upload, or numerical admission fails. An
    /// explicit NVIDIA, AMD, or Intel request returns an error instead of
    /// silently changing backend. Every GPU candidate must agree with scalar CPU
    /// inference on complete zero-input and start-position policy/WDL predictions
    /// before it is admitted.
    ///
    /// # Arguments
    ///
    /// * `path` - model container path
    /// * `backend` - execution backend selection policy
    /// * `device` - zero-based vendor-local device index
    ///
    /// # Returns
    ///
    /// Network whose backend status reports the admitted execution backend.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] for the same model failures as [`Self::load`]. An
    /// explicitly selected GPU also returns [`Bt4ErrorKind::Backend`] when its
    /// helper is unavailable, malformed, times out, or fails admission.
    pub fn load_with_backend(
        path: impl AsRef<Path>,
        backend: Bt4Backend,
        device: usize,
    ) -> Result<Self, Bt4Error> {
        let path = path.as_ref();
        let model = Bt4Model::open(path)?;
        if model.version() != 2 || !architecture_supported(model.architecture()) {
            return Err(Bt4Error::new(
                Bt4ErrorKind::UnsupportedShape,
                model.inference_unavailable_reason(),
            ));
        }
        let root_history = history_window("BT4 root history")?;
        let active_history = history_window("BT4 active history")?;
        let weights = load_weights(&model)?;
        let workspace = Workspace::new(&weights)?;
        let mut network = Self {
            weights,
            workspace,
            inference_threads: 1,
            root_history,
            active_history,
            #[cfg(feature = "gpu")]
            gpu: None,
            backend_status: Bt4BackendStatus::cpu(device),
        };
        network.configure_backend(path, backend, device)?;
        Ok(network)
    }

    /// Used for retrieving validated executable model metadata.
    #[must_use]
    pub const fn info(&self) -> Bt4Info {
        self.weights.info
    }

    /// Used for resolving and admitting the requested helper without
    /// changing CPU defaults.
    ///
    /// # Arguments
    ///
    /// * `model_path` - model container path forwarded to the helper
    /// * `backend` - execution backend selection policy
    /// * `device` - zero-based vendor-local device index
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when an explicitly requested GPU cannot start or
    /// fails admission; `Auto` failures fall back to CPU with a recorded
    /// reason instead.
    #[cfg(feature = "gpu")]
    fn configure_backend(
        &mut self,
        model_path: &Path,
        backend: Bt4Backend,
        device: usize,
    ) -> Result<(), Bt4Error> {
        if backend == Bt4Backend::Cpu {
            self.backend_status = Bt4BackendStatus::cpu(device);
            return Ok(());
        }
        debug_assert_eq!(GPU_INPUT_FLOATS, INPUT_CHANNELS * TOKENS);
        debug_assert_eq!(GPU_POLICY_FLOATS, INTERNAL_POLICY_SIZE);
        let mut client = match Bt4GpuClient::connect(model_path, backend, device) {
            Ok(client) => client,
            Err(error) => return self.handle_backend_start_failure(backend, device, &error),
        };
        if let Err(error) = self.admit_gpu(&mut client) {
            if backend == Bt4Backend::Auto {
                self.backend_status = Bt4BackendStatus::fallback(device, error.to_string());
                return Ok(());
            }
            return Err(error);
        }
        self.backend_status = client.status().clone();
        self.gpu = Some(client);
        Ok(())
    }

    /// Used for resolving the requested backend when the crate was compiled
    /// without the `gpu` feature.
    ///
    /// `Cpu` and `Auto` select the scalar CPU path, with `Auto` recording an
    /// observable fallback reason; an explicit GPU request fails because no
    /// helper client exists in this build.
    ///
    /// # Arguments
    ///
    /// * `model_path` - model container path, unused without a helper
    /// * `backend` - execution backend selection policy
    /// * `device` - zero-based vendor-local device index
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] with [`Bt4ErrorKind::Backend`] for an explicit
    /// NVIDIA, AMD, or Intel request.
    #[cfg(not(feature = "gpu"))]
    fn configure_backend(
        &mut self,
        model_path: &Path,
        backend: Bt4Backend,
        device: usize,
    ) -> Result<(), Bt4Error> {
        let _ = model_path;
        match backend {
            Bt4Backend::Cpu => {
                self.backend_status = Bt4BackendStatus::cpu(device);
                Ok(())
            }
            Bt4Backend::Auto => {
                self.backend_status = Bt4BackendStatus::fallback(
                    device,
                    "GPU support was not compiled into this build",
                );
                Ok(())
            }
            Bt4Backend::Nvidia | Bt4Backend::Amd | Bt4Backend::Intel => Err(Bt4Error::new(
                Bt4ErrorKind::Backend,
                format!("BT4 backend {backend} unavailable: GPU support was not compiled into this build"),
            )),
        }
    }

    /// Used for applying explicit-failure versus automatic-fallback startup
    /// semantics.
    ///
    /// # Arguments
    ///
    /// * `backend` - execution backend the caller requested
    /// * `device` - device index recorded for observability
    /// * `error` - helper startup failure
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] with [`Bt4ErrorKind::Backend`] for an explicit
    /// GPU request; an `Auto` request records a CPU fallback and succeeds.
    #[cfg(feature = "gpu")]
    fn handle_backend_start_failure(
        &mut self,
        backend: Bt4Backend,
        device: usize,
        error: &Bt4GpuError,
    ) -> Result<(), Bt4Error> {
        if backend == Bt4Backend::Auto {
            self.backend_status = Bt4BackendStatus::fallback(device, error.to_string());
            Ok(())
        } else {
            Err(backend_error(error))
        }
    }

    /// Used for requiring complete zero-input and start-position CPU/GPU
    /// predictions to agree within the declared FP32 cross-device tolerances
    /// before GPU use is reported active.
    ///
    /// Admission always runs single-threaded and clears the workspace before
    /// and after probing so no probe state leaks into later predictions.
    ///
    /// # Arguments
    ///
    /// * `client` - connected helper candidate under admission
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when probe allocation, encoding, GPU prediction,
    /// or the tolerance comparison fails.
    #[cfg(feature = "gpu")]
    fn admit_gpu(&mut self, client: &mut Bt4GpuClient) -> Result<(), Bt4Error> {
        let mut baseline_policy = zeroed(GPU_POLICY_FLOATS, "BT4 admission CPU policy")?;
        let mut candidate_policy = zeroed(GPU_POLICY_FLOATS, "BT4 admission GPU policy")?;
        let saved_threads = self.inference_threads;
        self.inference_threads = 1;
        let admission = (|| {
            self.workspace.clear();
            self.admit_gpu_workspace(client, &mut baseline_policy, &mut candidate_policy)?;

            self.workspace.clear();
            encode_lc0_fen_position_into(
                &Position::start(),
                self.weights.info.input_format,
                &mut self.workspace.encoded,
            )
            .map_err(|error| encoding_error(&error))?;
            self.admit_gpu_workspace(client, &mut baseline_policy, &mut candidate_policy)
        })();
        self.workspace.clear();
        self.inference_threads = saved_threads;
        admission
    }

    /// Used for comparing every policy and WDL output for the encoding
    /// currently in the workspace.
    ///
    /// # Arguments
    ///
    /// * `client` - connected helper candidate under admission
    /// * `baseline_policy` - destination for the scalar CPU policy
    /// * `candidate_policy` - destination for the helper policy
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when the GPU prediction fails or any output
    /// exceeds the admission tolerances.
    #[cfg(feature = "gpu")]
    fn admit_gpu_workspace(
        &mut self,
        client: &mut Bt4GpuClient,
        baseline_policy: &mut [f32],
        candidate_policy: &mut [f32],
    ) -> Result<(), Bt4Error> {
        let baseline_wdl = self.run_cpu_workspace();
        baseline_policy.copy_from_slice(&self.workspace.internal_policy);
        let candidate_wdl = client
            .predict(&self.workspace.encoded, candidate_policy)
            .map_err(|error| backend_error(&error))?;
        validate_gpu_admission(
            baseline_policy,
            candidate_policy,
            baseline_wdl,
            candidate_wdl,
        )
    }

    /// Used for selecting one to sixteen deterministic scoped dense workers.
    ///
    /// Zero becomes one and larger values are capped. Small operations retain
    /// their scalar path even when a larger team is configured.
    ///
    /// # Arguments
    ///
    /// * `threads` - requested worker count, clamped to
    ///   `1..=MAX_INFERENCE_THREADS`
    pub fn set_inference_threads(&mut self, threads: usize) {
        self.inference_threads = threads.clamp(1, MAX_INFERENCE_THREADS);
    }

    /// Used for retrieving the configured maximum worker count.
    #[must_use]
    pub const fn inference_threads(&self) -> usize {
        self.inference_threads
    }

    /// Used for retrieving the requested and currently active execution
    /// backend.
    #[must_use]
    pub const fn backend_status(&self) -> &Bt4BackendStatus {
        &self.backend_status
    }

    /// Used for retaining the newest eight real game positions ending at an
    /// MCTS root.
    ///
    /// Passing an empty slice clears temporal context. A search path whose root
    /// differs from the final retained position also falls back to the supplied
    /// root, preventing stale UCI history from leaking between games.
    ///
    /// # Arguments
    ///
    /// * `history_oldest_to_newest` - real game positions ending at the root
    pub fn set_root_history(&mut self, history_oldest_to_newest: &[Position]) {
        self.root_history.clear();
        let first = history_oldest_to_newest.len().saturating_sub(HISTORY);
        self.root_history
            .extend(history_oldest_to_newest[first..].iter().cloned());
        self.active_history.clear();
    }

    /// Used for evaluating caller-provided canonical channel-major
    /// `[112][64]` planes.
    ///
    /// # Arguments
    ///
    /// * `encoded` - exactly 7,168 finite channel-major input floats
    ///
    /// # Returns
    ///
    /// Complete compressed policy plus WDL prediction.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when the input length is not exactly 7,168, any
    /// input value is non-finite, or the returned policy allocation fails.
    pub fn predict_encoded(&mut self, encoded: &[f32]) -> Result<Bt4Prediction, Bt4Error> {
        let wdl = self.run_encoded(encoded)?;
        self.owned_prediction(wdl)
    }

    /// Used for evaluating one FEN-derived position with LC0's en-passant
    /// predecessor reconstruction.
    ///
    /// # Arguments
    ///
    /// * `position` - position to encode and evaluate
    ///
    /// # Returns
    ///
    /// Complete compressed policy plus WDL prediction.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] if encoding or the prediction allocation fails.
    pub fn predict_position(&mut self, position: &Position) -> Result<Bt4Prediction, Bt4Error> {
        encode_lc0_fen_position_into(
            position,
            self.weights.info.input_format,
            &mut self.workspace.encoded,
        )
        .map_err(|error| encoding_error(&error))?;
        let wdl = self.run_workspace();
        self.owned_prediction(wdl)
    }

    /// Used for evaluating an oldest-to-current position history.
    ///
    /// Missing slots repeat the oldest supplied position. Call
    /// [`Self::predict_position`] for a FEN-only position so the immediately
    /// preceding double pawn push can be reconstructed when possible.
    ///
    /// # Arguments
    ///
    /// * `history_oldest_to_newest` - positions ending at the one evaluated
    ///
    /// # Returns
    ///
    /// Complete compressed policy plus WDL prediction.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] for an empty history or prediction allocation
    /// failure.
    pub fn predict_history(
        &mut self,
        history_oldest_to_newest: &[Position],
    ) -> Result<Bt4Prediction, Bt4Error> {
        encode_history_into(
            history_oldest_to_newest,
            self.weights.info.input_format,
            &mut self.workspace.encoded,
        )
        .map_err(|error| encoding_error(&error))?;
        let wdl = self.run_workspace();
        self.owned_prediction(wdl)
    }

    /// Used for evaluating one position and extracting logits for only its
    /// legal moves.
    ///
    /// This is the normal MCTS entry point. It avoids allocating or copying the
    /// full 1,858-entry compressed policy.
    ///
    /// # Arguments
    ///
    /// * `position` - position to encode and evaluate
    /// * `legal_moves` - legal moves whose logits are gathered
    ///
    /// # Returns
    ///
    /// Shared MCTS policy/value output built from the WDL head and the
    /// gathered legal-move logits.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] if encoding or legal-policy gathering fails.
    pub fn predict_policy_value(
        &mut self,
        position: &Position,
        legal_moves: &[Move],
    ) -> Result<PolicyValue, Bt4Error> {
        let transform = encode_lc0_fen_position_into(
            position,
            self.weights.info.input_format,
            &mut self.workspace.encoded,
        )
        .map_err(|error| encoding_error(&error))?;
        let wdl = self.run_workspace();
        self.policy_value(position, legal_moves, transform, wdl)
    }

    /// Used for validating encoded planes and running the shared body plus
    /// both heads.
    ///
    /// # Arguments
    ///
    /// * `encoded` - exactly 7,168 finite channel-major input floats
    ///
    /// # Returns
    ///
    /// Win/draw/loss probabilities; the internal policy stays in the
    /// workspace.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] for a wrong input length or a non-finite input
    /// value.
    fn run_encoded(&mut self, encoded: &[f32]) -> Result<[f32; WDL_OUTPUTS], Bt4Error> {
        if encoded.len() != INPUT_CHANNELS * TOKENS {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!(
                    "encoded BT4 input has {} floats; expected {}",
                    encoded.len(),
                    INPUT_CHANNELS * TOKENS
                ),
            ));
        }
        if encoded.iter().any(|value| !value.is_finite()) {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                "encoded BT4 input contains a non-finite value",
            ));
        }
        self.workspace.encoded.copy_from_slice(encoded);
        Ok(self.run_workspace())
    }

    /// Used for running both heads from an already-populated encoded-input
    /// workspace.
    ///
    /// An admitted GPU serves the prediction first; any GPU failure drops the
    /// helper, records a runtime fallback reason, and retries on the scalar
    /// CPU path.
    ///
    /// # Returns
    ///
    /// Win/draw/loss probabilities; the internal policy stays in the
    /// workspace.
    fn run_workspace(&mut self) -> [f32; WDL_OUTPUTS] {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            let prediction = {
                let gpu = self.gpu.as_mut().expect("GPU presence was checked");
                gpu.predict(&self.workspace.encoded, &mut self.workspace.internal_policy)
            };
            match prediction {
                Ok(wdl) => return wdl,
                Err(error) => {
                    self.gpu.take();
                    self.backend_status = Bt4BackendStatus::runtime_fallback(
                        self.backend_status.requested(),
                        self.backend_status.device_index(),
                        format!("GPU inference failed: {error}"),
                    );
                }
            }
        }
        self.run_cpu_workspace()
    }

    /// Used for running the scalar body and both heads from the populated
    /// encoded workspace.
    ///
    /// # Returns
    ///
    /// Win/draw/loss probabilities; the internal policy stays in the
    /// workspace.
    fn run_cpu_workspace(&mut self) -> [f32; WDL_OUTPUTS] {
        run_body(&self.weights, &mut self.workspace, self.inference_threads);
        run_policy(&self.weights, &mut self.workspace, self.inference_threads);
        run_value(&self.weights, &mut self.workspace, self.inference_threads)
    }

    /// Used for copying the geometry-compressed policy into an owned
    /// prediction.
    ///
    /// # Arguments
    ///
    /// * `wdl` - win/draw/loss probabilities from the value head
    ///
    /// # Returns
    ///
    /// Prediction with the gathered compressed policy and the value computed
    /// as win minus loss.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when the policy allocation fails.
    fn owned_prediction(&self, wdl: [f32; WDL_OUTPUTS]) -> Result<Bt4Prediction, Bt4Error> {
        let mut policy = Vec::new();
        policy.try_reserve_exact(POLICY_SIZE).map_err(|_| {
            Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                "cannot allocate BT4 policy prediction",
            )
        })?;
        for &internal in &self.weights.policy_gather {
            policy.push(self.workspace.internal_policy[internal]);
        }
        Ok(Bt4Prediction {
            policy,
            wdl,
            value: wdl[0] - wdl[2],
        })
    }

    /// Used for gathering legal internal logits and constructing the shared
    /// MCTS value type.
    ///
    /// # Arguments
    ///
    /// * `position` - position the logits belong to
    /// * `legal_moves` - legal moves whose logits are gathered
    /// * `transform` - board transform returned by the encoder
    /// * `wdl` - win/draw/loss probabilities from the value head
    ///
    /// # Returns
    ///
    /// Policy/value output carrying win-minus-loss, draw probability, and the
    /// gathered legal-move logits.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] when legal-policy gathering fails.
    fn policy_value(
        &self,
        position: &Position,
        legal_moves: &[Move],
        transform: u8,
        wdl: [f32; WDL_OUTPUTS],
    ) -> Result<PolicyValue, Bt4Error> {
        let logits = gather_legal_internal_logits(
            position,
            legal_moves,
            &self.workspace.internal_policy,
            transform,
        )
        .map_err(|error| encoding_error(&error))?;
        Ok(PolicyValue::with_logits(wdl[0] - wdl[2], wdl[1], logits))
    }

    /// Used for evaluating with the temporal path prepared through the MCTS
    /// evaluator hooks.
    ///
    /// When the active history does not end at the supplied position, the
    /// single-position FEN path is used instead.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `legal_moves` - legal moves whose logits are gathered
    ///
    /// # Returns
    ///
    /// Shared MCTS policy/value output.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4Error`] if encoding or legal-policy gathering fails.
    fn predict_active_policy_value(
        &mut self,
        position: &Position,
        legal_moves: &[Move],
    ) -> Result<PolicyValue, Bt4Error> {
        if self.active_history.last() != Some(position) {
            return self.predict_policy_value(position, legal_moves);
        }
        let transform = encode_history_into(
            &self.active_history,
            self.weights.info.input_format,
            &mut self.workspace.encoded,
        )
        .map_err(|error| encoding_error(&error))?;
        let wdl = self.run_workspace();
        self.policy_value(position, legal_moves, transform, wdl)
    }

    /// Used for resetting the active MCTS window to a matching configured
    /// root history.
    ///
    /// A root that does not match the retained history starts a fresh
    /// single-position window instead.
    ///
    /// # Arguments
    ///
    /// * `root` - root position of the next MCTS simulation
    fn begin_path(&mut self, root: &Position) {
        self.active_history.clear();
        if self.root_history.last() == Some(root) {
            self.active_history
                .extend(self.root_history.iter().cloned());
        } else {
            self.active_history.push(root.clone());
        }
    }

    /// Used for appending one descendant while retaining the network's
    /// eight-slot window.
    ///
    /// A repeated final position is ignored and the oldest entry is dropped
    /// once the window is full.
    ///
    /// # Arguments
    ///
    /// * `position` - position selected by the current MCTS descent
    fn append_path_position(&mut self, position: &Position) {
        if self.active_history.last() == Some(position) {
            return;
        }
        if self.active_history.len() == HISTORY {
            self.active_history.remove(0);
        }
        self.active_history.push(position.clone());
    }
}

/// Used for bounding the absolute policy difference accepted across scalar
/// CPU and `OpenCL` arithmetic.
#[cfg(any(test, feature = "gpu"))]
const GPU_POLICY_ABSOLUTE_TOLERANCE: f32 = 2.0e-2;
/// Used for bounding the magnitude-scaled policy difference accepted in
/// addition to the absolute term.
#[cfg(any(test, feature = "gpu"))]
const GPU_POLICY_RELATIVE_TOLERANCE: f32 = 5.0e-3;
/// Used for bounding the absolute WDL probability difference accepted during
/// GPU admission.
#[cfg(any(test, feature = "gpu"))]
const GPU_WDL_ABSOLUTE_TOLERANCE: f32 = 2.0e-3;

/// Used for validating one complete CPU/GPU admission pair without hiding a
/// single output.
///
/// # Arguments
///
/// * `baseline_policy` - scalar CPU internal policy of full internal width
/// * `candidate_policy` - helper internal policy of full internal width
/// * `baseline_wdl` - scalar CPU win/draw/loss probabilities
/// * `candidate_wdl` - helper win/draw/loss probabilities
///
/// # Errors
///
/// Returns [`Bt4Error`] with [`Bt4ErrorKind::Backend`] for a wrong-sized
/// policy, a non-finite value, or any element outside the declared
/// tolerances.
#[cfg(any(test, feature = "gpu"))]
fn validate_gpu_admission(
    baseline_policy: &[f32],
    candidate_policy: &[f32],
    baseline_wdl: [f32; WDL_OUTPUTS],
    candidate_wdl: [f32; WDL_OUTPUTS],
) -> Result<(), Bt4Error> {
    if baseline_policy.len() != INTERNAL_POLICY_SIZE
        || candidate_policy.len() != INTERNAL_POLICY_SIZE
    {
        return Err(Bt4Error::new(
            Bt4ErrorKind::Backend,
            "BT4 GPU admission received a wrong-sized policy",
        ));
    }
    for (index, (&baseline, &candidate)) in baseline_policy.iter().zip(candidate_policy).enumerate()
    {
        let tolerance =
            GPU_POLICY_ABSOLUTE_TOLERANCE + GPU_POLICY_RELATIVE_TOLERANCE * baseline.abs();
        if !baseline.is_finite()
            || !candidate.is_finite()
            || (baseline - candidate).abs() > tolerance
        {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Backend,
                format!(
                    "BT4 GPU admission policy mismatch at {index}: CPU={baseline}, GPU={candidate}, tolerance={tolerance}"
                ),
            ));
        }
    }
    for (index, (&baseline, &candidate)) in baseline_wdl.iter().zip(&candidate_wdl).enumerate() {
        if !baseline.is_finite()
            || !candidate.is_finite()
            || (baseline - candidate).abs() > GPU_WDL_ABSOLUTE_TOLERANCE
        {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Backend,
                format!(
                    "BT4 GPU admission WDL mismatch at {index}: CPU={baseline}, GPU={candidate}, tolerance={GPU_WDL_ABSOLUTE_TOLERANCE}"
                ),
            ));
        }
    }
    Ok(())
}

impl Evaluator for Bt4Network {
    /// Used for converting the network WDL expectation to a bounded logistic
    /// score.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Logistic centipawn score, or zero when prediction fails.
    fn evaluate(&mut self, position: &Position) -> i32 {
        self.predict_position(position)
            .map_or(0, |prediction| value_to_centipawns(prediction.value))
    }

    /// Used for returning full BT4 policy/value output for MCTS expansion.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `legal_moves` - legal moves whose logits are gathered
    ///
    /// # Returns
    ///
    /// Policy/value output, or an empty neutral output when prediction
    /// fails.
    fn evaluate_policy_value(&mut self, position: &Position, legal_moves: &[Move]) -> PolicyValue {
        self.predict_active_policy_value(position, legal_moves)
            .unwrap_or_else(|_| PolicyValue::with_logits(0.0, 0.0, Vec::new()))
    }

    /// Used for rebuilding the temporal window for one deterministic MCTS
    /// simulation.
    ///
    /// # Arguments
    ///
    /// * `root` - root position of the simulation
    fn begin_mcts_path(&mut self, root: &Position) {
        self.begin_path(root);
    }

    /// Used for adding one selected descendant to the temporal network
    /// input.
    ///
    /// # Arguments
    ///
    /// * `position` - position selected by the current descent
    fn mcts_path_position(&mut self, position: &Position) {
        self.append_path_position(position);
    }

    /// Used for preventing a second full-network call inside recursive MCTS
    /// quiescence.
    ///
    /// # Returns
    ///
    /// Always `false`.
    fn allows_mcts_quiescence(&self) -> bool {
        false
    }
}

/// Used for converting an encoder contract error into the model's public
/// error family.
///
/// # Arguments
///
/// * `error` - encoding failure to convert
///
/// # Returns
///
/// Format-kind [`Bt4Error`] carrying the encoder message.
fn encoding_error(error: &Bt4EncodingError) -> Bt4Error {
    Bt4Error::new(Bt4ErrorKind::Format, error.to_string())
}

/// Used for converting a fault-isolated helper failure to the public BT4
/// error family.
///
/// # Arguments
///
/// * `error` - helper failure to convert
///
/// # Returns
///
/// Backend-kind [`Bt4Error`] carrying the helper diagnostic.
#[cfg(feature = "gpu")]
fn backend_error(error: &Bt4GpuError) -> Bt4Error {
    Bt4Error::new(Bt4ErrorKind::Backend, error.to_string())
}

/// Used for mapping a bounded WDL expectation to a finite logistic centipawn
/// score.
///
/// The expectation is clamped just inside `(-1, 1)` before the logistic
/// transform so the score stays finite.
///
/// # Arguments
///
/// * `value` - expected game result in `[-1, 1]`
///
/// # Returns
///
/// Rounded logistic centipawn score.
#[allow(clippy::cast_possible_truncation)]
fn value_to_centipawns(value: f32) -> i32 {
    let bounded = f64::from(value).clamp(-0.999_999, 0.999_999);
    (300.0 * ((1.0 + bounded) / (1.0 - bounded)).ln()).round() as i32
}

/// Used for moving the validated manifest payloads into the pinned
/// executable layout.
///
/// Tensors are consumed in exact manifest order: the input stack, fifteen
/// encoder blocks, the shared smolgen matrix, and the policy and value heads.
/// The `DeepNet` residual scale is derived from the encoder-layer count.
///
/// # Arguments
///
/// * `model` - opened and structurally validated BT4 container
///
/// # Returns
///
/// Complete typed parameter set including the geometry-derived policy
/// gather.
///
/// # Errors
///
/// Returns [`Bt4Error`] when tensor decoding, any name or shape check, the
/// encoder-table reservation, or the policy-gather construction fails.
#[allow(clippy::too_many_lines)]
fn load_weights(model: &Bt4Model) -> Result<Weights, Bt4Error> {
    let payloads = model.load_all_tensors()?;
    let mut cursor = TensorCursor::new(model.tensors(), payloads)?;
    let preproc = cursor.dense("input.preproc", PREPROC_CHANNELS * TOKENS, 512 * TOKENS)?;
    let embedding = cursor.dense("input.embedding", INPUT_CHANNELS + 512, EMBEDDING)?;
    let embedding_ln_gamma = cursor.tensor("input.embedding_ln_gamma", EMBEDDING)?;
    let embedding_ln_beta = cursor.tensor("input.embedding_ln_beta", EMBEDDING)?;
    let mult_gate = token_to_feature_major(
        &cursor.tensor("input.mult_gate", TOKENS * EMBEDDING)?,
        EMBEDDING,
        TOKENS,
    )?;
    let add_gate = token_to_feature_major(
        &cursor.tensor("input.add_gate", TOKENS * EMBEDDING)?,
        EMBEDDING,
        TOKENS,
    )?;
    let input_ffn_in = cursor.dense("input.ffn.in", EMBEDDING, FFN_HIDDEN)?;
    let input_ffn_out = cursor.dense("input.ffn.out", FFN_HIDDEN, EMBEDDING)?;
    let input_ffn_ln_gamma = cursor.tensor("input.ffn_ln_gamma", EMBEDDING)?;
    let input_ffn_ln_beta = cursor.tensor("input.ffn_ln_beta", EMBEDDING)?;
    let input = InputStack {
        preproc,
        embedding,
        embedding_ln_gamma,
        embedding_ln_beta,
        mult_gate,
        add_gate,
        ffn_in: input_ffn_in,
        ffn_out: input_ffn_out,
        ffn_ln_gamma: input_ffn_ln_gamma,
        ffn_ln_beta: input_ffn_ln_beta,
    };

    let mut encoders = Vec::new();
    encoders.try_reserve_exact(ENCODER_LAYERS).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            "cannot reserve BT4 encoder table",
        )
    })?;
    for block in 0..ENCODER_LAYERS {
        let prefix = format!("body.encoder[{block}]");
        let query = cursor.dense(&format!("{prefix}.attention.query"), EMBEDDING, EMBEDDING)?;
        let key = cursor.dense(&format!("{prefix}.attention.key"), EMBEDDING, EMBEDDING)?;
        let value = cursor.dense(&format!("{prefix}.attention.value"), EMBEDDING, EMBEDDING)?;
        let output = cursor.dense(&format!("{prefix}.attention.out"), EMBEDDING, EMBEDDING)?;
        let compress = cursor.dense(
            &format!("{prefix}.smolgen.compress"),
            EMBEDDING,
            SMOLGEN_CHANNELS,
        )?;
        let dense1 = cursor.dense(
            &format!("{prefix}.smolgen.dense1"),
            SMOLGEN_CHANNELS * TOKENS,
            SMOLGEN_HIDDEN,
        )?;
        let ln1_gamma = cursor.tensor(&format!("{prefix}.smolgen.ln1_gamma"), SMOLGEN_HIDDEN)?;
        let ln1_beta = cursor.tensor(&format!("{prefix}.smolgen.ln1_beta"), SMOLGEN_HIDDEN)?;
        let dense2 = cursor.dense(
            &format!("{prefix}.smolgen.dense2"),
            SMOLGEN_HIDDEN,
            ATTENTION_HEADS * SMOLGEN_PER_HEAD,
        )?;
        let ln2_gamma = cursor.tensor(
            &format!("{prefix}.smolgen.ln2_gamma"),
            ATTENTION_HEADS * SMOLGEN_PER_HEAD,
        )?;
        let ln2_beta = cursor.tensor(
            &format!("{prefix}.smolgen.ln2_beta"),
            ATTENTION_HEADS * SMOLGEN_PER_HEAD,
        )?;
        let ffn_in = cursor.dense(&format!("{prefix}.ffn.in"), EMBEDDING, FFN_HIDDEN)?;
        let ffn_out = cursor.dense(&format!("{prefix}.ffn.out"), FFN_HIDDEN, EMBEDDING)?;
        let body_ln1_gamma = cursor.tensor(&format!("{prefix}.ln1_gamma"), EMBEDDING)?;
        let body_ln1_beta = cursor.tensor(&format!("{prefix}.ln1_beta"), EMBEDDING)?;
        let body_ln2_gamma = cursor.tensor(&format!("{prefix}.ln2_gamma"), EMBEDDING)?;
        let body_ln2_beta = cursor.tensor(&format!("{prefix}.ln2_beta"), EMBEDDING)?;
        encoders.push(EncoderBlock {
            attention: Attention {
                query,
                key,
                value,
                output,
                smolgen: Smolgen {
                    compress,
                    dense1,
                    ln1_gamma,
                    ln1_beta,
                    dense2,
                    ln2_gamma,
                    ln2_beta,
                },
            },
            ffn_in,
            ffn_out,
            ln1_gamma: body_ln1_gamma,
            ln1_beta: body_ln1_beta,
            ln2_gamma: body_ln2_gamma,
            ln2_beta: body_ln2_beta,
        });
    }
    let smolgen_projection = cursor.tensor("body.smolgen_w", SMOLGEN_PER_HEAD * ATTENTION_MAP)?;
    let policy = PolicyHead {
        embedding: cursor.dense("policy.embedding", EMBEDDING, EMBEDDING)?,
        query: cursor.dense("policy.query", EMBEDDING, EMBEDDING)?,
        key: cursor.dense("policy.key", EMBEDDING, EMBEDDING)?,
        promotion_weights: cursor.tensor("policy.promotion_weights", 4 * EMBEDDING)?,
    };
    let value = ValueHead {
        embedding: cursor.dense("value.embedding", EMBEDDING, 128)?,
        fc1: cursor.dense("value.fc1", 128 * TOKENS, 128)?,
        fc2: cursor.dense("value.fc2", 128, WDL_OUTPUTS)?,
    };
    cursor.finish()?;
    let policy_gather = build_policy_gather()?;
    let residual_alpha = (2.0_f64 * ENCODER_LAYERS as f64).powf(-0.25) as f32;
    Ok(Weights {
        info: Bt4Info {
            input_format: model.architecture().input_format,
            input_channels: INPUT_CHANNELS,
            tokens: TOKENS,
            embedding_size: EMBEDDING,
            encoder_layers: ENCODER_LAYERS,
            attention_heads: ATTENTION_HEADS,
            policy_size: POLICY_SIZE,
            parameter_count: model.parameter_count(),
        },
        input,
        encoders,
        smolgen_projection,
        policy,
        value,
        policy_gather,
        layer_norm_epsilon: model.architecture().layer_norm_epsilon,
        residual_alpha,
    })
}

/// Used for converting a serialized token-major gate to the activation
/// layout once at load time.
///
/// # Arguments
///
/// * `token_major` - serialized `[token][feature]` values
/// * `features` - feature count
/// * `tokens` - token count
///
/// # Returns
///
/// Transposed `[feature][token]` copy.
///
/// # Errors
///
/// Returns [`Bt4Error`] when the destination allocation fails.
fn token_to_feature_major(
    token_major: &[f32],
    features: usize,
    tokens: usize,
) -> Result<Vec<f32>, Bt4Error> {
    debug_assert_eq!(token_major.len(), features * tokens);
    let mut feature_major = zeroed(token_major.len(), "transposed BT4 gate")?;
    for token in 0..tokens {
        for feature in 0..features {
            feature_major[feature * tokens + token] = token_major[token * features + feature];
        }
    }
    Ok(feature_major)
}

/// Used for fallibly creating a zero-filled inference buffer.
///
/// # Arguments
///
/// * `elements` - float count to reserve exactly
/// * `label` - buffer name used in the failure diagnostic
///
/// # Returns
///
/// Zero-filled buffer of the requested length.
///
/// # Errors
///
/// Returns [`Bt4Error`] with [`Bt4ErrorKind::ResourceLimit`] when the
/// reservation fails.
fn zeroed(elements: usize, label: &str) -> Result<Vec<f32>, Bt4Error> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            format!("cannot reserve {elements} floats for {label}"),
        )
    })?;
    values.resize(elements, 0.0);
    Ok(values)
}

/// Used for fallibly admitting one bounded BT4 position-history window.
///
/// The complete eight-position capacity is reserved before a network is
/// published, so later bounded `extend` and `push` operations cannot encounter
/// the allocator after model admission.
///
/// # Arguments
///
/// * `label` - stable ownership name included in a refusal diagnostic
///
/// # Returns
///
/// Empty position vector with capacity for exactly [`HISTORY`] entries.
///
/// # Errors
///
/// Returns [`Bt4ErrorKind::ResourceLimit`] when the allocator refuses the
/// fixed history capacity.
fn history_window(label: &str) -> Result<Vec<Position>, Bt4Error> {
    history_window_with_capacity(HISTORY, label)
}

/// Used for testing and implementing fallible BT4 history reservation.
///
/// Production calls pass [`HISTORY`]; accepting the capacity explicitly lets
/// focused tests exercise arithmetic refusal without exhausting host memory.
///
/// # Arguments
///
/// * `capacity` - exact number of positions to reserve
/// * `label` - stable ownership name included in a refusal diagnostic
///
/// # Returns
///
/// Empty position vector with at least the requested capacity.
///
/// # Errors
///
/// Returns [`Bt4ErrorKind::ResourceLimit`] when capacity overflows or the
/// allocator refuses it.
fn history_window_with_capacity(capacity: usize, label: &str) -> Result<Vec<Position>, Bt4Error> {
    let mut history = Vec::new();
    history.try_reserve_exact(capacity).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            format!("cannot reserve {capacity} positions for {label}"),
        )
    })?;
    Ok(history)
}

/// Used for running `PE_DENSE`, the gated input FFN, and all transformer
/// body blocks.
///
/// The final body activation is left in `workspace.flow` for both heads.
///
/// # Arguments
///
/// * `weights` - complete typed parameter set
/// * `workspace` - buffers with `encoded` already populated
/// * `threads` - configured maximum dense-worker count
fn run_body(weights: &Weights, workspace: &mut Workspace, threads: usize) {
    for token in 0..TOKENS {
        for channel in 0..PREPROC_CHANNELS {
            workspace.preproc_input[token * PREPROC_CHANNELS + channel] =
                workspace.encoded[channel * TOKENS + token];
        }
    }
    weights.input.preproc.forward(
        &workspace.preproc_input,
        1,
        &mut workspace.preproc_output,
        threads,
    );
    let preprocessed_per_token = weights.input.preproc.output / TOKENS;
    for channel in 0..INPUT_CHANNELS {
        let base = channel * TOKENS;
        workspace.embedding_input[base..base + TOKENS]
            .copy_from_slice(&workspace.encoded[base..base + TOKENS]);
    }
    for feature in 0..preprocessed_per_token {
        let output_base = (INPUT_CHANNELS + feature) * TOKENS;
        for token in 0..TOKENS {
            workspace.embedding_input[output_base + token] =
                workspace.preproc_output[token * preprocessed_per_token + feature];
        }
    }
    weights.input.embedding.forward(
        &workspace.embedding_input,
        TOKENS,
        &mut workspace.flow,
        threads,
    );
    activate_in_place(&mut workspace.flow, Bt4Activation::Mish);
    layer_norm_feature_major(
        &mut workspace.flow,
        TOKENS,
        EMBEDDING,
        &weights.input.embedding_ln_gamma,
        &weights.input.embedding_ln_beta,
        weights.layer_norm_epsilon,
    );
    for index in 0..workspace.flow.len() {
        workspace.flow[index] *= weights.input.mult_gate[index];
        workspace.flow[index] += weights.input.add_gate[index];
    }
    weights
        .input
        .ffn_in
        .forward(&workspace.flow, TOKENS, &mut workspace.ffn_hidden, threads);
    activate_in_place(&mut workspace.ffn_hidden, Bt4Activation::Mish);
    weights
        .input
        .ffn_out
        .forward(&workspace.ffn_hidden, TOKENS, &mut workspace.next, threads);
    residual_add_in_place(&mut workspace.next, &workspace.flow, weights.residual_alpha);
    layer_norm_feature_major(
        &mut workspace.next,
        TOKENS,
        EMBEDDING,
        &weights.input.ffn_ln_gamma,
        &weights.input.ffn_ln_beta,
        weights.layer_norm_epsilon,
    );
    workspace.flow.copy_from_slice(&workspace.next);

    for block in &weights.encoders {
        run_attention(
            block,
            &weights.smolgen_projection,
            weights.layer_norm_epsilon,
            workspace,
            threads,
        );
        residual_add_in_place(&mut workspace.next, &workspace.flow, weights.residual_alpha);
        layer_norm_feature_major(
            &mut workspace.next,
            TOKENS,
            EMBEDDING,
            &block.ln1_gamma,
            &block.ln1_beta,
            weights.layer_norm_epsilon,
        );
        block
            .ffn_in
            .forward(&workspace.next, TOKENS, &mut workspace.ffn_hidden, threads);
        activate_in_place(&mut workspace.ffn_hidden, Bt4Activation::Mish);
        block
            .ffn_out
            .forward(&workspace.ffn_hidden, TOKENS, &mut workspace.flow, threads);
        residual_add_in_place(&mut workspace.flow, &workspace.next, weights.residual_alpha);
        layer_norm_feature_major(
            &mut workspace.flow,
            TOKENS,
            EMBEDDING,
            &block.ln2_gamma,
            &block.ln2_beta,
            weights.layer_norm_epsilon,
        );
    }
}

/// Used for running smolgen-biased multi-head attention and writing its
/// output projection to `workspace.next`.
///
/// # Arguments
///
/// * `block` - encoder block supplying the attention parameters
/// * `shared_smolgen` - globally shared smolgen projection matrix
/// * `epsilon` - layer-normalization epsilon
/// * `workspace` - buffers with `flow` holding the block input
/// * `threads` - configured maximum dense-worker count
fn run_attention(
    block: &EncoderBlock,
    shared_smolgen: &[f32],
    epsilon: f32,
    workspace: &mut Workspace,
    threads: usize,
) {
    block
        .attention
        .query
        .forward(&workspace.flow, TOKENS, &mut workspace.query, threads);
    block
        .attention
        .key
        .forward(&workspace.flow, TOKENS, &mut workspace.key, threads);
    block
        .attention
        .value
        .forward(&workspace.flow, TOKENS, &mut workspace.value, threads);
    compute_smolgen_bias(
        &workspace.flow,
        &block.attention.smolgen,
        shared_smolgen,
        epsilon,
        &mut workspace.smolgen_compressed,
        &mut workspace.smolgen_flat,
        &mut workspace.smolgen_mid,
        &mut workspace.smolgen_generated,
        &mut workspace.smolgen_bias,
        threads,
    );
    attention_heads(
        &workspace.query,
        &workspace.key,
        &workspace.value,
        &workspace.smolgen_bias,
        &mut workspace.combined,
        threads,
    );
    block
        .attention
        .output
        .forward(&workspace.combined, TOKENS, &mut workspace.next, threads);
}

/// Used for computing one block's complete `[head][query][key]` smolgen
/// bias.
///
/// # Arguments
///
/// * `input` - feature-major body activation
/// * `smolgen` - per-block smolgen parameters
/// * `shared` - globally shared smolgen projection matrix
/// * `epsilon` - layer-normalization epsilon
/// * `compressed` - scratch for the feature-major compression output
/// * `flat` - scratch for the token-major flattened compression
/// * `mid` - scratch for the first vector projection
/// * `generated` - scratch for the per-head coefficients
/// * `bias` - destination for the per-head attention biases
/// * `threads` - configured maximum dense-worker count
#[allow(clippy::too_many_arguments)]
fn compute_smolgen_bias(
    input: &[f32],
    smolgen: &Smolgen,
    shared: &[f32],
    epsilon: f32,
    compressed: &mut [f32],
    flat: &mut [f32],
    mid: &mut [f32],
    generated: &mut [f32],
    bias: &mut [f32],
    threads: usize,
) {
    smolgen.compress.forward(input, TOKENS, compressed, threads);
    for token in 0..TOKENS {
        for channel in 0..SMOLGEN_CHANNELS {
            flat[token * SMOLGEN_CHANNELS + channel] = compressed[channel * TOKENS + token];
        }
    }
    smolgen.dense1.forward(flat, 1, mid, threads);
    activate_in_place(mid, Bt4Activation::Swish);
    layer_norm_feature_major(
        mid,
        1,
        SMOLGEN_HIDDEN,
        &smolgen.ln1_gamma,
        &smolgen.ln1_beta,
        epsilon,
    );
    smolgen.dense2.forward(mid, 1, generated, threads);
    activate_in_place(generated, Bt4Activation::Swish);
    layer_norm_feature_major(
        generated,
        1,
        ATTENTION_HEADS * SMOLGEN_PER_HEAD,
        &smolgen.ln2_gamma,
        &smolgen.ln2_beta,
        epsilon,
    );
    project_smolgen(shared, generated, bias, threads);
}

/// Used for projecting independent per-head coefficient vectors through the
/// globally shared row-major smolgen matrix.
///
/// Heads are independent, so the projection may split across scoped workers
/// at head boundaries without changing any scalar result.
///
/// # Arguments
///
/// * `shared` - shared `[attention_cell][per_head_feature]` matrix
/// * `generated` - per-head coefficient vectors
/// * `bias` - destination `[head][attention_cell]` biases
/// * `threads` - configured maximum worker count
fn project_smolgen(shared: &[f32], generated: &[f32], bias: &mut [f32], threads: usize) {
    debug_assert_eq!(shared.len(), ATTENTION_MAP * SMOLGEN_PER_HEAD);
    debug_assert_eq!(generated.len(), ATTENTION_HEADS * SMOLGEN_PER_HEAD);
    debug_assert_eq!(bias.len(), ATTENTION_HEADS * ATTENTION_MAP);
    let workers = threads.min(ATTENTION_HEADS);
    if workers <= 1 {
        project_smolgen_heads(shared, generated, bias, 0);
        return;
    }
    let heads_per_worker = ATTENTION_HEADS.div_ceil(workers);
    let values_per_worker = heads_per_worker * ATTENTION_MAP;
    std::thread::scope(|scope| {
        let mut chunks = bias.chunks_mut(values_per_worker).enumerate();
        let (_, caller_output) = chunks.next().expect("smolgen bias is nonempty");
        for (chunk_index, output_chunk) in chunks {
            let first_head = chunk_index * heads_per_worker;
            spawn_scoped_or_run(scope, "janus-bt4-smolgen", move || {
                project_smolgen_heads(shared, generated, output_chunk, first_head);
            });
        }
        project_smolgen_heads(shared, generated, caller_output, 0);
    });
}

/// Used for projecting a contiguous worker-owned range of smolgen heads.
///
/// # Arguments
///
/// * `shared` - shared `[attention_cell][per_head_feature]` matrix
/// * `generated` - per-head coefficient vectors for all heads
/// * `output` - destination slice covering whole heads
/// * `first_head` - global index of the first head written
fn project_smolgen_heads(shared: &[f32], generated: &[f32], output: &mut [f32], first_head: usize) {
    debug_assert_eq!(output.len() % ATTENTION_MAP, 0);
    for (local_head, output_head) in output.chunks_exact_mut(ATTENTION_MAP).enumerate() {
        let generated_base = (first_head + local_head) * SMOLGEN_PER_HEAD;
        for (cell, output_value) in output_head.iter_mut().enumerate() {
            let weight_base = cell * SMOLGEN_PER_HEAD;
            let mut sum = 0.0_f32;
            for dimension in 0..SMOLGEN_PER_HEAD {
                sum += generated[generated_base + dimension] * shared[weight_base + dimension];
            }
            *output_value = sum;
        }
    }
}

/// Used for computing all attention heads, splitting only at head boundaries
/// so every softmax and value reduction remains single-worker and
/// deterministic.
///
/// # Arguments
///
/// * `query` - feature-major query projection
/// * `key` - feature-major key projection
/// * `value` - feature-major value projection
/// * `bias` - per-head `[head][query][key]` smolgen biases
/// * `output` - feature-major concatenated head destination
/// * `threads` - configured maximum worker count
fn attention_heads(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    bias: &[f32],
    output: &mut [f32],
    threads: usize,
) {
    let workers = threads.min(ATTENTION_HEADS);
    if workers <= 1 {
        attention_head_range(query, key, value, bias, output, 0);
        return;
    }
    let depth = EMBEDDING / ATTENTION_HEADS;
    let heads_per_worker = ATTENTION_HEADS.div_ceil(workers);
    let features_per_worker = heads_per_worker * depth;
    let values_per_worker = features_per_worker * TOKENS;
    std::thread::scope(|scope| {
        let mut chunks = output.chunks_mut(values_per_worker).enumerate();
        let (_, caller_output) = chunks.next().expect("attention output is nonempty");
        for (chunk_index, output_chunk) in chunks {
            let first_head = chunk_index * heads_per_worker;
            spawn_scoped_or_run(scope, "janus-bt4-attention", move || {
                attention_head_range(query, key, value, bias, output_chunk, first_head);
            });
        }
        attention_head_range(query, key, value, bias, caller_output, 0);
    });
}

/// Used for computing a contiguous range of heads into its local
/// feature-major slice.
///
/// Each head applies scaled query-key scores plus its smolgen bias, a stable
/// softmax, and the value reduction for every query token.
///
/// # Arguments
///
/// * `query` - feature-major query projection
/// * `key` - feature-major key projection
/// * `value` - feature-major value projection
/// * `bias` - per-head `[head][query][key]` smolgen biases
/// * `output` - destination slice covering whole heads
/// * `first_head` - global index of the first head written
fn attention_head_range(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    bias: &[f32],
    output: &mut [f32],
    first_head: usize,
) {
    let depth = EMBEDDING / ATTENTION_HEADS;
    debug_assert_eq!(output.len() % (depth * TOKENS), 0);
    let local_heads = output.len() / (depth * TOKENS);
    let inverse_scale = (1.0_f64 / (depth as f64).sqrt()) as f32;
    for local_head in 0..local_heads {
        let head = first_head + local_head;
        let feature_base = head * depth;
        let local_feature_base = local_head * depth;
        let bias_base = head * ATTENTION_MAP;
        for query_token in 0..TOKENS {
            let mut scores = [0.0_f32; TOKENS];
            for key_token in 0..TOKENS {
                let mut sum = 0.0_f32;
                for dimension in 0..depth {
                    let feature = feature_base + dimension;
                    sum +=
                        query[feature * TOKENS + query_token] * key[feature * TOKENS + key_token];
                }
                scores[key_token] =
                    sum * inverse_scale + bias[bias_base + query_token * TOKENS + key_token];
            }
            softmax_in_place(&mut scores);
            for dimension in 0..depth {
                let feature = feature_base + dimension;
                let mut sum = 0.0_f32;
                for key_token in 0..TOKENS {
                    sum += scores[key_token] * value[feature * TOKENS + key_token];
                }
                output[(local_feature_base + dimension) * TOKENS + query_token] = sum;
            }
        }
    }
}

/// Used for running the attention policy head into the reusable internal
/// policy tensor.
///
/// Scaled from/to query-key products fill the 64x64 portion; the promotion
/// planes are appended afterwards.
///
/// # Arguments
///
/// * `weights` - complete typed parameter set
/// * `workspace` - buffers with `flow` holding the final body activation
/// * `threads` - configured maximum dense-worker count
fn run_policy(weights: &Weights, workspace: &mut Workspace, threads: usize) {
    weights
        .policy
        .embedding
        .forward(&workspace.flow, TOKENS, &mut workspace.policy_flow, threads);
    activate_in_place(&mut workspace.policy_flow, Bt4Activation::Mish);
    weights.policy.query.forward(
        &workspace.policy_flow,
        TOKENS,
        &mut workspace.policy_query,
        threads,
    );
    weights.policy.key.forward(
        &workspace.policy_flow,
        TOKENS,
        &mut workspace.policy_key,
        threads,
    );
    workspace.internal_policy.fill(0.0);
    let policy_dimension = weights.policy.query.output;
    let inverse_scale = (1.0_f64 / (policy_dimension as f64).sqrt()) as f32;
    for from in 0..TOKENS {
        for to in 0..TOKENS {
            let mut sum = 0.0_f32;
            for dimension in 0..policy_dimension {
                sum += workspace.policy_query[dimension * TOKENS + from]
                    * workspace.policy_key[dimension * TOKENS + to];
            }
            workspace.internal_policy[from * TOKENS + to] = sum * inverse_scale;
        }
    }
    add_promotion_logits(
        &workspace.policy_key,
        policy_dimension,
        &weights.policy.promotion_weights,
        &mut workspace.internal_policy,
    );
}

/// Used for adding the three promotion-offset planes exactly as LC0's
/// attention backend.
///
/// Each row combines the from/to knight baseline, one piece offset, and the
/// fourth shared knight-offset projection.
///
/// # Arguments
///
/// * `key` - feature-major policy key projection
/// * `policy_dimension` - policy projection width
/// * `promotion_weights` - four promotion projections in row-major order
/// * `internal` - internal policy tensor receiving the promotion planes
fn add_promotion_logits(
    key: &[f32],
    policy_dimension: usize,
    promotion_weights: &[f32],
    internal: &mut [f32],
) {
    for from_file in 0_usize..8 {
        let minimum_to = from_file.saturating_sub(1);
        let maximum_to = (from_file + 1).min(7);
        for to_file in minimum_to..=maximum_to {
            let from = 48 + from_file;
            let to = 56 + to_file;
            let base = internal[from * TOKENS + to];
            let shared_knight =
                promotion_projection(key, to, policy_dimension, promotion_weights, 3);
            for promotion in 0..3 {
                let index = FROM_TO_POLICY_SIZE + from_file * 24 + to_file * 3 + promotion;
                internal[index] = base
                    + shared_knight
                    + promotion_projection(key, to, policy_dimension, promotion_weights, promotion);
            }
        }
    }
}

/// Used for computing one destination-key promotion projection.
///
/// # Arguments
///
/// * `key` - feature-major policy key projection
/// * `token` - destination square token
/// * `policy_dimension` - policy projection width
/// * `weights` - four promotion projections in row-major order
/// * `output` - promotion projection row index
///
/// # Returns
///
/// Dot product of the destination key and the selected projection row.
fn promotion_projection(
    key: &[f32],
    token: usize,
    policy_dimension: usize,
    weights: &[f32],
    output: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    let weight_base = output * policy_dimension;
    for dimension in 0..policy_dimension {
        sum += key[dimension * TOKENS + token] * weights[weight_base + dimension];
    }
    sum
}

/// Used for running the WDL head and returning normalized win/draw/loss
/// probabilities.
///
/// # Arguments
///
/// * `weights` - complete typed parameter set
/// * `workspace` - buffers with `flow` holding the final body activation
/// * `threads` - configured maximum dense-worker count
///
/// # Returns
///
/// Softmax-normalized probabilities ordered win, draw, loss.
fn run_value(weights: &Weights, workspace: &mut Workspace, threads: usize) -> [f32; WDL_OUTPUTS] {
    weights.value.embedding.forward(
        &workspace.flow,
        TOKENS,
        &mut workspace.value_embedding,
        threads,
    );
    activate_in_place(&mut workspace.value_embedding, Bt4Activation::Mish);
    let value_features = weights.value.embedding.output;
    for token in 0..TOKENS {
        for feature in 0..value_features {
            workspace.value_flat[token * value_features + feature] =
                workspace.value_embedding[feature * TOKENS + token];
        }
    }
    weights.value.fc1.forward(
        &workspace.value_flat,
        1,
        &mut workspace.value_hidden,
        threads,
    );
    activate_in_place(&mut workspace.value_hidden, Bt4Activation::Mish);
    weights.value.fc2.forward(
        &workspace.value_hidden,
        1,
        &mut workspace.value_logits,
        threads,
    );
    softmax_in_place(&mut workspace.value_logits);
    [
        workspace.value_logits[0],
        workspace.value_logits[1],
        workspace.value_logits[2],
    ]
}

/// Used for applying an activation with the Java reference's f64
/// transcendental and f32 storage boundaries.
///
/// # Arguments
///
/// * `values` - activations rewritten in place
/// * `activation` - activation kind; `None` leaves the values untouched
fn activate_in_place(values: &mut [f32], activation: Bt4Activation) {
    if matches!(activation, Bt4Activation::None) {
        return;
    }
    for value in values {
        let input = *value;
        *value = match activation {
            Bt4Activation::None => input,
            Bt4Activation::Relu => input.max(0.0),
            Bt4Activation::Mish => input * f64::from(softplus(input)).tanh() as f32,
            Bt4Activation::Swish => input / (1.0 + (-f64::from(input)).exp() as f32),
            Bt4Activation::Tanh => f64::from(input).tanh() as f32,
        };
    }
}

/// Used for computing the numerically stable softplus used by Mish.
///
/// Large positive inputs return themselves and large negative inputs return
/// the plain exponential, avoiding overflow in `ln(1 + e^x)`.
///
/// # Arguments
///
/// * `value` - softplus input
///
/// # Returns
///
/// Softplus of the input at the reference precision boundary.
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        f64::from(value).exp() as f32
    } else {
        f64::from(value).exp().ln_1p() as f32
    }
}

/// Used for applying layer normalization independently to every token of a
/// feature-major activation.
///
/// # Arguments
///
/// * `values` - feature-major `[feature][token]` activations rewritten in
///   place
/// * `tokens` - token count
/// * `features` - feature count normalized per token
/// * `gamma` - per-feature scale vector
/// * `beta` - per-feature bias vector
/// * `epsilon` - variance stabilizer added before the square root
fn layer_norm_feature_major(
    values: &mut [f32],
    tokens: usize,
    features: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    debug_assert_eq!(values.len(), features * tokens);
    debug_assert_eq!(gamma.len(), features);
    debug_assert_eq!(beta.len(), features);
    for token in 0..tokens {
        let mut mean = 0.0_f32;
        for feature in 0..features {
            mean += values[feature * tokens + token];
        }
        mean /= features as f32;
        let mut variance = 0.0_f32;
        for feature in 0..features {
            let centered = values[feature * tokens + token] - mean;
            variance += centered * centered;
        }
        let denominator = variance / features as f32 + epsilon;
        let inverse_standard_deviation = (1.0_f64 / f64::from(denominator).sqrt()) as f32;
        for feature in 0..features {
            let index = feature * tokens + token;
            values[index] = (values[index] - mean) * inverse_standard_deviation * gamma[feature]
                + beta[feature];
        }
    }
}

/// Used for scaling a sublayer output before adding the matching residual
/// activation.
///
/// # Arguments
///
/// * `output` - sublayer output scaled by `alpha` and rewritten in place
/// * `residual` - residual activation added element-wise
/// * `alpha` - `DeepNet` residual scale
fn residual_add_in_place(output: &mut [f32], residual: &[f32], alpha: f32) {
    debug_assert_eq!(output.len(), residual.len());
    for index in 0..output.len() {
        output[index] *= alpha;
        output[index] += residual[index];
    }
}

/// Used for applying an in-place stable softmax using the reference f64
/// exponential and f32 accumulation behavior.
///
/// The maximum is subtracted before exponentiation; an all-zero sum leaves
/// the exponentials unnormalized instead of dividing by zero.
///
/// # Arguments
///
/// * `values` - logits rewritten in place with their softmax
fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = f64::from(*value - maximum).exp() as f32;
        sum += *value;
    }
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

/// Used for building compressed-policy order directly from queen/knight
/// geometry and the three legal promotion-offset destination files.
///
/// # Returns
///
/// Exactly [`POLICY_SIZE`] internal indices in compressed LC0 policy order.
///
/// # Errors
///
/// Returns [`Bt4Error`] when the reservation fails or the generated geometry
/// does not produce exactly [`POLICY_SIZE`] entries.
fn build_policy_gather() -> Result<Vec<usize>, Bt4Error> {
    let mut gather = Vec::new();
    gather.try_reserve_exact(POLICY_SIZE).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            "cannot reserve BT4 policy gather",
        )
    })?;
    for from in 0..TOKENS {
        for to in 0..TOKENS {
            if from != to && is_queen_like_or_knight(from, to) {
                gather.push(from * TOKENS + to);
            }
        }
    }
    for from_file in 0_usize..8 {
        let minimum_to = from_file.saturating_sub(1);
        let maximum_to = (from_file + 1).min(7);
        for to_file in minimum_to..=maximum_to {
            for promotion in 0..3 {
                gather.push(FROM_TO_POLICY_SIZE + from_file * 24 + to_file * 3 + promotion);
            }
        }
    }
    if gather.len() != POLICY_SIZE {
        return Err(Bt4Error::new(
            Bt4ErrorKind::UnsupportedShape,
            format!(
                "BT4 geometry produced {} policy entries; expected {POLICY_SIZE}",
                gather.len()
            ),
        ));
    }
    Ok(gather)
}

/// Used for checking whether a square pair is represented by the attention
/// policy.
///
/// # Arguments
///
/// * `from` - origin square index in `0..64`
/// * `to` - destination square index in `0..64`
///
/// # Returns
///
/// `true` for rook-, bishop-, or knight-reachable square pairs; a
/// same-square pair also matches and is excluded by the caller.
fn is_queen_like_or_knight(from: usize, to: usize) -> bool {
    let from_file = i32::try_from(from % 8).expect("board file fits i32");
    let from_rank = i32::try_from(from / 8).expect("board rank fits i32");
    let to_file = i32::try_from(to % 8).expect("board file fits i32");
    let to_rank = i32::try_from(to / 8).expect("board rank fits i32");
    let file_delta = to_file - from_file;
    let rank_delta = to_rank - from_rank;
    file_delta == 0
        || rank_delta == 0
        || file_delta.abs() == rank_delta.abs()
        || KNIGHT_DELTAS.contains(&(file_delta, rank_delta))
}

