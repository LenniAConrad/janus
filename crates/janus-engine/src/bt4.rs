//! Safe, dependency-free loading of CRTK's compact BT4 model format.
//!
//! CRTK BT4 files are large (the repository model is about 706 MiB), so this
//! module parses their tensor layout lazily.  Opening a model validates every
//! dimension, tensor length, cross-layer shape, and the final file boundary
//! without retaining hundreds of megabytes of weights.  Individual tensors
//! can then be loaded by index, or all payloads can be streamed through the
//! numeric validator.
//!
//! The manifest remains useful independently of inference.  The executable
//! backend lives in the sibling `bt4_inference` module and consumes the same
//! checked tensor manifest, so model inspection and execution cannot disagree
//! about tensor boundaries or resource limits.

use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Used for identifying a CRTK BT4 container by its four-byte signature.
///
/// The structural parser rejects any file whose first four bytes are not
/// exactly `b"BT4J"`.
const MAGIC: [u8; 4] = *b"BT4J";
/// Used for bounding the oldest container layout understood by the
/// structural parser.
///
/// Versions below this value are rejected with
/// [`Bt4ErrorKind::UnsupportedVersion`].
const MIN_VERSION: u32 = 1;
/// Used for bounding the newest container layout understood by the
/// structural parser.
///
/// Versions above this value are rejected with
/// [`Bt4ErrorKind::UnsupportedVersion`].
const MAX_VERSION: u32 = 2;
/// Used for capping the byte length of any length-prefixed header string.
///
/// [`Parser::string`] rejects longer declared lengths before allocating the
/// string buffer.
const MAX_STRING_BYTES: u32 = 1_000_000;
/// Used for capping the value of one architecture dimension.
///
/// [`Parser::dimension`] and [`checked_add`] enforce this bound so later
/// element-count arithmetic cannot silently overflow.
const MAX_DIMENSION: u32 = 65_536;
/// Used for capping the number of encoder blocks accepted in one sequence.
///
/// Both the main transformer body and the policy-head encoder sequence are
/// counted against this limit through [`Parser::count`].
const MAX_ENCODER_BLOCKS: u32 = 256;
/// Used for capping the number of `f32` values accepted in one tensor.
///
/// [`Parser::array`] and [`checked_mul`] enforce this bound before any tensor
/// range is recorded in the manifest.
const MAX_TENSOR_ELEMENTS: u64 = 100_000_000;
/// Used for bounding the size of a complete BT4 container accepted by the
/// dependency-free loader.
///
/// [`Bt4Model::open`] rejects larger files before any parsing begins.
pub const MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;
/// Used for bounding the aggregate parameter count over all tensors in one
/// BT4 container.
///
/// Derived from [`MAX_MODEL_BYTES`] divided by the four bytes occupied by one
/// serialized `f32` value.
pub const MAX_MODEL_PARAMETERS: u64 = MAX_MODEL_BYTES / 4;
/// Used for sizing the reusable byte buffer employed by finite checked tensor
/// decoding.
///
/// Both [`load_tensor_values`] and [`Bt4Model::validate_numeric_weights`]
/// stream payload bytes through a buffer of this size, keeping auxiliary
/// memory constant regardless of tensor size.
const TENSOR_READ_BUFFER_BYTES: usize = 64 * 1024;

/// Broad category for a BT4 load failure.
///
/// Every [`Bt4Error`] carries exactly one of these kinds so callers can
/// distinguish I/O trouble, malformed bytes, and deliberate safety-limit
/// rejections without parsing diagnostic text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bt4ErrorKind {
    /// Used for reporting that a filesystem or read operation failed.
    Io,
    /// Used for reporting that the byte stream is truncated or otherwise
    /// malformed.
    Format,
    /// Used for reporting a container version outside the supported v1/v2
    /// range.
    UnsupportedVersion,
    /// Used for reporting tensor dimensions that do not implement the
    /// architecture they advertise.
    UnsupportedShape,
    /// Used for reporting a file, allocation, or aggregate parameter count
    /// that exceeds a fixed bound.
    ResourceLimit,
    /// Used for reporting that an explicitly requested GPU helper could not
    /// start or execute correctly.
    Backend,
}

/// Error returned while parsing or loading a CRTK BT4 file.
///
/// Combines a stable [`Bt4ErrorKind`] category with a human-readable message
/// and, for filesystem failures, the underlying [`io::Error`] exposed through
/// the standard `source` chain.
#[derive(Debug)]
pub struct Bt4Error {
    /// Used for storing the stable category suitable for programmatic
    /// handling.
    kind: Bt4ErrorKind,
    /// Used for storing human-readable context for the failed field or
    /// operation.
    message: String,
    /// Used for storing the underlying I/O error when the failure originated
    /// in the filesystem.
    source: Option<io::Error>,
}

impl Bt4Error {
    /// Used for creating a format or architecture error without an underlying
    /// source.
    ///
    /// # Arguments
    ///
    /// * `kind` - stable error category
    /// * `message` - human-readable context for the failed field or operation
    ///
    /// # Returns
    ///
    /// Error carrying the category and message with no source error.
    pub(crate) fn new(kind: Bt4ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// Used for wrapping an I/O error while preserving a task-specific
    /// diagnostic.
    ///
    /// # Arguments
    ///
    /// * `message` - task-specific context for the failed operation
    /// * `source` - underlying filesystem or read error
    ///
    /// # Returns
    ///
    /// Error of kind [`Bt4ErrorKind::Io`] chaining the source error.
    fn io(message: impl Into<String>, source: io::Error) -> Self {
        Self {
            kind: Bt4ErrorKind::Io,
            message: message.into(),
            source: Some(source),
        }
    }

    /// Used for retrieving the stable error category.
    ///
    /// # Returns
    ///
    /// The [`Bt4ErrorKind`] recorded when the error was constructed.
    #[must_use]
    pub const fn kind(&self) -> Bt4ErrorKind {
        self.kind
    }
}

impl fmt::Display for Bt4Error {
    /// Used for writing the context-rich diagnostic without duplicating the
    /// source text.
    ///
    /// The wrapped I/O error, when present, remains reachable through
    /// `std::error::Error::source` instead of being repeated here.
    ///
    /// # Arguments
    ///
    /// * `f` - destination formatter
    ///
    /// # Returns
    ///
    /// Propagated formatter result.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Bt4Error {
    /// Used for exposing the wrapped I/O failure, when one exists.
    ///
    /// # Returns
    ///
    /// The underlying [`io::Error`] for I/O failures, otherwise `None`.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

/// LC0 input-plane interpretation recorded in a BT4 model.
///
/// The header stores this choice as a string; the parser accepts exactly the
/// three spellings listed on the variants and rejects everything else.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bt4InputFormat {
    /// Used for selecting the classical 112 LC0 planes with the legacy
    /// auxiliary-plane semantics (`CLASSICAL_112`).
    Classical112,
    /// Used for selecting the classical 112 planes whose castling state is
    /// encoded explicitly (`CASTLING_PLANE_112`).
    CastlingPlane112,
    /// Used for selecting BT4's canonical side-to-move-oriented 112-plane
    /// representation (`BT4_CANONICAL_112`).
    Canonical112,
}

/// Position embedding strategy recorded in a BT4 model.
///
/// The header stores this choice as a string; `PE_DENSE` is only accepted for
/// version-2 containers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bt4InputEmbedding {
    /// Used for indicating that no learned positional representation is
    /// appended or projected (`NONE`).
    None,
    /// Used for indicating that a one-hot square-identity position map is
    /// appended to every input token (`PE_MAP`).
    PositionMap,
    /// Used for indicating the v2 dense positional projection and
    /// normalization tensors (`PE_DENSE`).
    PositionDense,
}

/// Activation names accepted by the Java BT4 reference implementation.
///
/// `Parser::activation` maps the serialized uppercase spellings onto these
/// variants and rejects any other name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bt4Activation {
    /// Used for leaving the layer output linear (`NONE`).
    None,
    /// Used for applying the rectified linear unit (`RELU`).
    Relu,
    /// Used for applying the Mish activation used by some LC0 transformer
    /// networks (`MISH`).
    Mish,
    /// Used for applying the sigmoid-weighted linear unit, SiLU/Swish
    /// (`SWISH`).
    Swish,
    /// Used for applying the hyperbolic tangent (`TANH`).
    Tanh,
}

/// Version-2 architecture extensions.
///
/// The four booleans mirror independent flag bytes in the on-disk v2 header;
/// combining them into an enum would permit invalid or incomplete states.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, PartialEq)]
pub struct Bt4V2Extensions {
    /// Used for storing the width of the feed-forward sublayer in each
    /// encoder block.
    pub ffn_hidden_size: u32,
    /// Used for storing the channel count produced by the per-token smolgen
    /// compression layer.
    pub smolgen_hidden_channels: u32,
    /// Used for storing the width of the first dense smolgen projection.
    pub smolgen_hidden_size: u32,
    /// Used for storing the number of smolgen coefficients generated for each
    /// attention head.
    pub smolgen_per_head_dim: u32,
    /// Used for storing the shared smolgen matrix area; files with smolgen
    /// enabled must set this to `tokens * tokens`.
    pub smolgen_global_size: u32,
    /// Used for storing the default activation recorded for v2 layers without
    /// a specific override.
    pub default_activation: Bt4Activation,
    /// Used for storing the activation used by smolgen projections.
    pub smolgen_activation: Bt4Activation,
    /// Used for storing the activation used between feed-forward dense
    /// layers.
    pub ffn_activation: Bt4Activation,
    /// Used for recording whether the input contains the v2 dense
    /// preprocessing projection.
    pub has_input_preproc: bool,
    /// Used for recording whether the input embedding is followed by an
    /// additional FFN block.
    pub has_input_embedding_ffn: bool,
    /// Used for recording whether multiplicative and additive input-gate
    /// tensors are present.
    pub has_input_gates: bool,
    /// Used for recording whether encoder blocks contain smolgen tensors.
    pub has_smolgen: bool,
}

/// Architecture metadata shared by the v1 and v2 CRTK BT4 containers.
///
/// All fields are validated by `Parser::parse_header` before any tensor is
/// accepted, so a constructed value always describes a self-consistent
/// header.
#[derive(Clone, Debug, PartialEq)]
pub struct Bt4Architecture {
    /// Used for storing the human-readable architecture identifier stored by
    /// the exporter.
    pub name: String,
    /// Used for storing the semantics of the input plane tensor.
    pub input_format: Bt4InputFormat,
    /// Used for storing the positional embedding strategy applied before the
    /// encoder body.
    pub input_embedding: Bt4InputEmbedding,
    /// Used for storing the number of scalar features supplied for each board
    /// token.
    pub input_channels: u32,
    /// Used for storing the token count; current chess networks normally use
    /// 64 board squares.
    pub tokens: u32,
    /// Used for storing the width of every token in the main transformer
    /// body.
    pub embedding_size: u32,
    /// Used for storing the number of encoder blocks in the main transformer
    /// body.
    pub encoder_layers: u32,
    /// Used for storing the number of attention heads in each main-body
    /// encoder block.
    pub attention_heads: u32,
    /// Used for storing the policy output width declared by the container.
    pub policy_size: u32,
    /// Used for storing the positive epsilon used by layer normalization.
    pub layer_norm_epsilon: f32,
    /// Used for storing the version-2-only dimensions and optional-block
    /// flags.
    pub v2: Option<Bt4V2Extensions>,
}

/// Location of one length-prefixed float32 tensor inside the model file.
///
/// Entries are produced in serialization order by the structural parser and
/// describe exactly where each payload begins and how many values it holds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bt4TensorInfo {
    /// Used for storing the stable hierarchical name, including block
    /// indices.
    pub name: String,
    /// Used for storing the number of little-endian float32 values.
    pub elements: u64,
    /// Used for storing the byte offset of the first float (after the length
    /// prefix).
    pub byte_offset: u64,
}

/// A structurally validated, lazily loaded BT4 model file.
///
/// Opening a model records only the architecture header and the tensor
/// manifest; weight payloads stay on disk until explicitly requested through
/// [`Bt4Model::load_tensor`], `Bt4Model::load_all_tensors`, or the
/// streaming numeric validator.
#[derive(Debug)]
pub struct Bt4Model {
    /// Used for storing the path reopened by lazy tensor loading and numeric
    /// validation.
    path: PathBuf,
    /// Used for storing the validated container version.
    version: u32,
    /// Used for storing the parsed architecture header.
    architecture: Bt4Architecture,
    /// Used for storing the ordered tensor locations established by the
    /// structural pass.
    tensors: Vec<Bt4TensorInfo>,
    /// Used for storing the checked sum of all tensor element counts.
    parameter_count: u64,
    /// Used for storing the file length captured by the structural pass.
    file_bytes: u64,
}

impl Bt4Model {
    /// Used for opening a v1 or v2 CRTK BT4 file and validating its complete
    /// tensor layout.
    ///
    /// Weight payloads are seeked over rather than retained, keeping peak
    /// memory independent of model size.  Use [`Self::load_tensor`] or
    /// [`Self::validate_numeric_weights`] when payload bytes must be read.
    ///
    /// # Arguments
    ///
    /// * `path` - filesystem location of the BT4 container
    ///
    /// # Returns
    ///
    /// Validated model with its architecture header and tensor manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, is not a v1/v2 CRTK BT4
    /// container, contains an invalid or unsupported tensor layout, or
    /// exceeds a file-size or parameter-count safety limit.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Bt4Error> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| {
            Bt4Error::io(format!("cannot open BT4 model {}", path.display()), error)
        })?;
        let file_bytes = file
            .metadata()
            .map_err(|error| Bt4Error::io("cannot read BT4 model metadata", error))?
            .len();
        if file_bytes > MAX_MODEL_BYTES {
            return Err(Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                format!("BT4 model is {file_bytes} bytes; safety limit is {MAX_MODEL_BYTES}"),
            ));
        }
        let mut parser = Parser::new(file, file_bytes);
        let (version, architecture) = parser.parse_header()?;
        parser.parse_weights(version, &architecture)?;
        if parser.position != file_bytes {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!(
                    "unexpected trailing bytes in BT4 model: parsed {}, file has {}",
                    parser.position, file_bytes
                ),
            ));
        }
        if parser.parameter_count > MAX_MODEL_PARAMETERS {
            return Err(Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                format!(
                    "BT4 model has {} parameters; safety limit is {MAX_MODEL_PARAMETERS}",
                    parser.parameter_count
                ),
            ));
        }
        let mut owned_path = PathBuf::new();
        owned_path
            .try_reserve(path.as_os_str().len())
            .map_err(|_| {
                Bt4Error::new(
                    Bt4ErrorKind::ResourceLimit,
                    "cannot reserve the BT4 model path",
                )
            })?;
        owned_path.push(path);
        Ok(Self {
            path: owned_path,
            version,
            architecture,
            parameter_count: parser.parameter_count,
            tensors: parser.tensors,
            file_bytes,
        })
    }

    /// Used for retrieving the CRTK BT4 container version (1 or 2).
    ///
    /// Only versions inside the supported range survive [`Self::open`].
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Used for retrieving the parsed architecture metadata.
    ///
    /// # Returns
    ///
    /// Borrowed [`Bt4Architecture`] validated during [`Self::open`].
    #[must_use]
    pub const fn architecture(&self) -> &Bt4Architecture {
        &self.architecture
    }

    /// Used for retrieving the full lazy tensor manifest in file order.
    ///
    /// # Returns
    ///
    /// Borrowed slice of [`Bt4TensorInfo`] entries in serialization order.
    #[must_use]
    pub fn tensors(&self) -> &[Bt4TensorInfo] {
        &self.tensors
    }

    /// Used for retrieving the total number of float32 parameters in all
    /// tensors.
    ///
    /// The sum is overflow-checked during parsing and bounded by
    /// [`MAX_MODEL_PARAMETERS`].
    #[must_use]
    pub const fn parameter_count(&self) -> u64 {
        self.parameter_count
    }

    /// Used for retrieving the exact file size observed when the manifest was
    /// parsed.
    ///
    /// Later lazy reads compare the live file length against this value and
    /// refuse to touch a model that changed on disk.
    #[must_use]
    pub const fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// Used for loading one tensor by manifest index.
    ///
    /// The model file is reopened, checked against the recorded length, and
    /// the requested payload is decoded through the finite-value validator.
    ///
    /// # Arguments
    ///
    /// * `index` - zero-based position in the manifest returned by
    ///   [`Self::tensors`]
    ///
    /// # Returns
    ///
    /// Decoded float values of the requested tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid index, a file that changed after it was
    /// opened, an I/O failure, or a non-finite tensor value.
    pub fn load_tensor(&self, index: usize) -> Result<Vec<f32>, Bt4Error> {
        let tensor = self.tensors.get(index).ok_or_else(|| {
            Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("BT4 tensor index {index} is out of range"),
            )
        })?;
        let mut file = File::open(&self.path)
            .map_err(|error| Bt4Error::io("cannot reopen BT4 model", error))?;
        ensure_same_file_length(&file, self.file_bytes)?;
        let mut buffer = zeroed_bytes(TENSOR_READ_BUFFER_BYTES, "BT4 tensor read buffer")?;
        load_tensor_values(&mut file, tensor, &mut buffer)
    }

    /// Used for loading all tensors with one file handle and moving each
    /// payload into its final typed layer without retaining a second
    /// full-model byte buffer.
    ///
    /// # Returns
    ///
    /// One decoded float vector per manifest entry, in file order.
    ///
    /// # Errors
    ///
    /// Returns an error if the file changed, an allocation cannot be reserved,
    /// an I/O operation fails, or any decoded parameter is non-finite.
    pub(crate) fn load_all_tensors(&self) -> Result<Vec<Vec<f32>>, Bt4Error> {
        let mut file = File::open(&self.path)
            .map_err(|error| Bt4Error::io("cannot reopen BT4 model", error))?;
        ensure_same_file_length(&file, self.file_bytes)?;
        let mut tensors = Vec::new();
        tensors.try_reserve_exact(self.tensors.len()).map_err(|_| {
            Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                "cannot reserve BT4 tensor table",
            )
        })?;
        let mut buffer = zeroed_bytes(TENSOR_READ_BUFFER_BYTES, "BT4 tensor read buffer")?;
        for tensor in &self.tensors {
            tensors.push(load_tensor_values(&mut file, tensor, &mut buffer)?);
        }
        Ok(tensors)
    }

    /// Used for streaming every weight and rejecting NaN or infinite
    /// parameters.
    ///
    /// The structural parser does not read 706 MiB merely to open the standard
    /// model.  This explicit integrity gate performs that full numeric pass
    /// with a fixed-size buffer and constant auxiliary memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the file changed, cannot be read, or contains a NaN
    /// or infinite parameter.
    pub fn validate_numeric_weights(&self) -> Result<(), Bt4Error> {
        let mut file = File::open(&self.path)
            .map_err(|error| Bt4Error::io("cannot reopen BT4 model", error))?;
        ensure_same_file_length(&file, self.file_bytes)?;
        let mut buffer = zeroed_bytes(TENSOR_READ_BUFFER_BYTES, "BT4 tensor read buffer")?;
        let chunk_capacity = u64::try_from(buffer.len() / 4)
            .map_err(|_| shape("numeric validation buffer size overflows u64"))?;
        for tensor in &self.tensors {
            file.seek(SeekFrom::Start(tensor.byte_offset))
                .map_err(|error| Bt4Error::io("cannot seek to BT4 tensor", error))?;
            let mut remaining = tensor.elements;
            let mut index = 0_u64;
            while remaining != 0 {
                let values = usize::try_from(remaining.min(chunk_capacity))
                    .map_err(|_| shape("numeric validation chunk does not fit usize"))?;
                let bytes = values * 4;
                file.read_exact(&mut buffer[..bytes]).map_err(|error| {
                    Bt4Error::io(format!("cannot validate {}", tensor.name), error)
                })?;
                for chunk in buffer[..bytes].chunks_exact(4) {
                    let value = f32::from_bits(u32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]));
                    if !value.is_finite() {
                        return Err(Bt4Error::new(
                            Bt4ErrorKind::Format,
                            format!("{} contains a non-finite value at {index}", tensor.name),
                        ));
                    }
                    index += 1;
                }
                remaining -= u64::try_from(values)
                    .map_err(|_| shape("numeric validation chunk does not fit u64"))?;
            }
        }
        Ok(())
    }

    /// Used for reporting whether the executable backend accepts this exact
    /// architecture.
    ///
    /// # Returns
    ///
    /// `true` when `crate::bt4_inference` supports the parsed architecture.
    #[must_use]
    pub const fn scalar_inference_supported(&self) -> bool {
        crate::bt4_inference::architecture_supported(&self.architecture)
    }

    /// Used for explaining why a structurally valid non-pinned architecture is
    /// rejected.
    ///
    /// # Returns
    ///
    /// Static diagnostic naming the single supported inference architecture.
    #[must_use]
    pub const fn inference_unavailable_reason(&self) -> &'static str {
        "safe scalar inference supports the pinned full BT4 v2 CLASSICAL_112 1024x15x32h architecture only"
    }
}

/// Used for decoding one tensor in fixed-size chunks while checking allocation
/// and every floating-point value before it becomes executable model state.
///
/// The vector allocation goes through `try_reserve_exact`, so an oversized
/// tensor surfaces as a [`Bt4ErrorKind::ResourceLimit`] error rather than an
/// abort.
///
/// # Arguments
///
/// * `file` - open model file positioned anywhere; the tensor offset is
///   seeked explicitly
/// * `tensor` - manifest entry describing the payload location and length
/// * `buffer` - reusable read buffer whose length is a multiple of four
///
/// # Returns
///
/// Decoded finite float values of the tensor.
///
/// # Errors
///
/// Returns an error when the allocation cannot be reserved, an I/O operation
/// fails, or a decoded value is NaN or infinite.
///
/// # Panics
///
/// Panics in debug builds when `buffer` is not a whole number of four-byte
/// words long.
fn load_tensor_values(
    file: &mut File,
    tensor: &Bt4TensorInfo,
    buffer: &mut [u8],
) -> Result<Vec<f32>, Bt4Error> {
    debug_assert_eq!(buffer.len() % 4, 0);
    let count = usize::try_from(tensor.elements).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            format!("{} does not fit in addressable memory", tensor.name),
        )
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            format!("cannot reserve {} float values", tensor.name),
        )
    })?;
    file.seek(SeekFrom::Start(tensor.byte_offset))
        .map_err(|error| Bt4Error::io(format!("cannot seek to {}", tensor.name), error))?;
    let mut remaining = count;
    let mut element = 0_usize;
    while remaining != 0 {
        let chunk_values = remaining.min(buffer.len() / 4);
        let chunk_bytes = chunk_values * 4;
        file.read_exact(&mut buffer[..chunk_bytes])
            .map_err(|error| Bt4Error::io(format!("cannot read {}", tensor.name), error))?;
        for bytes in buffer[..chunk_bytes].chunks_exact(4) {
            let value =
                f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            if !value.is_finite() {
                return Err(Bt4Error::new(
                    Bt4ErrorKind::Format,
                    format!("{} contains a non-finite value at {element}", tensor.name),
                ));
            }
            values.push(value);
            element += 1;
        }
        remaining -= chunk_values;
    }
    Ok(values)
}

/// Used for fallibly creating a zero-filled byte buffer owned by the BT4
/// loader.
///
/// Reserving the complete capacity before initialization keeps bounded model
/// data and fixed decode scratch inside [`Bt4ErrorKind::ResourceLimit`] rather
/// than the workspace's aborting allocation path.
///
/// # Arguments
///
/// * `length` - exact byte length to reserve and initialize
/// * `label` - stable buffer name included in a refusal diagnostic
///
/// # Returns
///
/// Zero-filled byte vector of exactly `length` bytes.
///
/// # Errors
///
/// Returns [`Bt4ErrorKind::ResourceLimit`] when the capacity overflows or the
/// allocator refuses it.
fn zeroed_bytes(length: usize, label: &str) -> Result<Vec<u8>, Bt4Error> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| {
        Bt4Error::new(
            Bt4ErrorKind::ResourceLimit,
            format!("cannot reserve {length} bytes for {label}"),
        )
    })?;
    bytes.resize(length, 0);
    Ok(bytes)
}

/// Used for rejecting a model whose length changed after its manifest was
/// established.
///
/// Lazy tensor reads trust byte offsets recorded during the structural pass;
/// a resized file would silently invalidate them, so it is refused instead.
///
/// # Arguments
///
/// * `file` - freshly reopened model file
/// * `expected` - byte length captured by the structural pass
///
/// # Errors
///
/// Returns an error when the metadata cannot be read or the current length
/// differs from `expected`.
fn ensure_same_file_length(file: &File, expected: u64) -> Result<(), Bt4Error> {
    let actual = file
        .metadata()
        .map_err(|error| Bt4Error::io("cannot read BT4 model metadata", error))?
        .len();
    if actual != expected {
        return Err(Bt4Error::new(
            Bt4ErrorKind::Format,
            format!("BT4 model changed after parsing: {expected} bytes became {actual}"),
        ));
    }
    Ok(())
}

/// Input/output dimensions of one serialized row-major dense layer.
///
/// Produced by [`Parser::dense`] after the weight matrix and bias vector have
/// been recorded, and consumed by [`expect_dense`] shape checks.
#[derive(Clone, Copy, Debug)]
struct DenseShape {
    /// Used for storing the number of input values consumed per output row.
    input: u32,
    /// Used for storing the number of bias values and output rows.
    output: u32,
}

/// Streaming structural parser for a BT4 container.
///
/// `position` is the authoritative byte cursor and is advanced for both
/// header reads and tensor seeks. Each accepted tensor is recorded in file
/// order without loading its payload. Thus a successful parse establishes
/// that every manifest range is in-bounds, non-overlapping, and that the
/// final parsed field ends at the exact file boundary checked by
/// [`Bt4Model::open`].
struct Parser {
    /// Used for storing the direct file handle used for bounded header reads
    /// and large forward seeks without a hidden infallible heap buffer.
    input: File,
    /// Used for storing the file length captured before parsing.
    file_bytes: u64,
    /// Used for storing the offset immediately after the last consumed or
    /// skipped field.
    position: u64,
    /// Used for storing the tensor manifest accumulated in serialization
    /// order.
    tensors: Vec<Bt4TensorInfo>,
    /// Used for storing the checked sum of all tensor element counts.
    parameter_count: u64,
}

impl Parser {
    /// Used for creating a parser positioned at the first byte of `file`.
    ///
    /// # Arguments
    ///
    /// * `file` - open model file at offset zero
    /// * `file_bytes` - total file length captured before parsing
    ///
    /// # Returns
    ///
    /// Parser with an empty manifest and a zeroed byte cursor.
    fn new(file: File, file_bytes: u64) -> Self {
        Self {
            input: file,
            file_bytes,
            position: 0,
            tensors: Vec::new(),
            parameter_count: 0,
        }
    }

    /// Used for reading and validating the common container header and v2
    /// extension.
    ///
    /// Checks the magic signature, the supported version range, the input
    /// format and embedding vocabulary, dimension bounds, the positivity of
    /// the layer-norm epsilon, head divisibility, and — for version 2 — the
    /// extension flags including the `tokens * tokens` smolgen constraint.
    ///
    /// # Returns
    ///
    /// Validated container version paired with the parsed architecture.
    ///
    /// # Errors
    ///
    /// Returns an error for a bad magic value, an unsupported version, an
    /// unknown format, embedding, or activation string, an out-of-range
    /// dimension, or an inconsistent v2 extension header.
    fn parse_header(&mut self) -> Result<(u32, Bt4Architecture), Bt4Error> {
        let mut magic = [0_u8; 4];
        self.read_exact(&mut magic, "BT4 magic")?;
        if magic != MAGIC {
            return Err(Bt4Error::new(Bt4ErrorKind::Format, "invalid BT4 magic"));
        }
        let version = self.u32("BT4 version")?;
        if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
            return Err(Bt4Error::new(
                Bt4ErrorKind::UnsupportedVersion,
                format!("unsupported BT4 version {version}; expected 1 or 2"),
            ));
        }
        let name = self.string("architecture name")?;
        if name.trim().is_empty() {
            return Err(shape("architecture name is blank"));
        }
        let input_format = match self.string("input format")?.as_str() {
            "CLASSICAL_112" => Bt4InputFormat::Classical112,
            "CASTLING_PLANE_112" => Bt4InputFormat::CastlingPlane112,
            "BT4_CANONICAL_112" => Bt4InputFormat::Canonical112,
            other => return Err(shape(format!("unsupported BT4 input format {other}"))),
        };
        let input_embedding = match self.string("input embedding")?.as_str() {
            "NONE" => Bt4InputEmbedding::None,
            "PE_MAP" => Bt4InputEmbedding::PositionMap,
            "PE_DENSE" if version == 2 => Bt4InputEmbedding::PositionDense,
            other => return Err(shape(format!("unsupported BT4 input embedding {other}"))),
        };
        let input_channels = self.dimension("input channels")?;
        let tokens = self.dimension("tokens")?;
        let embedding_size = self.dimension("embedding size")?;
        let encoder_layers = self.count("encoder layers", MAX_ENCODER_BLOCKS)?;
        let attention_heads = self.dimension("attention heads")?;
        let policy_size = self.dimension("policy size")?;
        let layer_norm_epsilon = self.f32("layer norm epsilon")?;
        if !layer_norm_epsilon.is_finite() || layer_norm_epsilon <= 0.0 {
            return Err(shape("layer norm epsilon must be finite and positive"));
        }
        if embedding_size % attention_heads != 0 {
            return Err(shape("embedding size is not divisible by attention heads"));
        }
        let v2 = if version == 2 {
            let ffn_hidden_size = self.dimension("FFN hidden size")?;
            let smolgen_hidden_channels = self.dimension("smolgen hidden channels")?;
            let smolgen_hidden_size = self.dimension("smolgen hidden size")?;
            let smolgen_per_head_dim = self.dimension("smolgen per-head dimension")?;
            let smolgen_global_size = self.dimension("smolgen global size")?;
            let default_activation = self.activation("default activation")?;
            let smolgen_activation = self.activation("smolgen activation")?;
            let ffn_activation = self.activation("FFN activation")?;
            let has_input_preproc = self.boolean("has input preproc")?;
            let has_input_embedding_ffn = self.boolean("has input embedding FFN")?;
            let has_input_gates = self.boolean("has input gates")?;
            let has_smolgen = self.boolean("has smolgen")?;
            if input_embedding == Bt4InputEmbedding::PositionDense && !has_input_preproc {
                return Err(shape("PE_DENSE requires input preprocessing"));
            }
            if has_smolgen {
                let expected = checked_mul(tokens, tokens, "tokens squared")?;
                if smolgen_global_size != expected {
                    return Err(shape("smolgen global size must equal tokens squared"));
                }
            }
            Some(Bt4V2Extensions {
                ffn_hidden_size,
                smolgen_hidden_channels,
                smolgen_hidden_size,
                smolgen_per_head_dim,
                smolgen_global_size,
                default_activation,
                smolgen_activation,
                ffn_activation,
                has_input_preproc,
                has_input_embedding_ffn,
                has_input_gates,
                has_smolgen,
            })
        } else {
            None
        };
        Ok((
            version,
            Bt4Architecture {
                name,
                input_format,
                input_embedding,
                input_channels,
                tokens,
                embedding_size,
                encoder_layers,
                attention_heads,
                policy_size,
                layer_norm_epsilon,
                v2,
            },
        ))
    }

    /// Used for walking the version-specific tensor grammar and validating
    /// cross-layer shapes.
    ///
    /// Version 1 reads a single input embedding sized for the optional
    /// position map; version 2 delegates to [`Self::parse_v2_input`]. Both
    /// versions then parse the encoder body (with the v2 residual-alpha
    /// constant when applicable), the optional shared smolgen tensor, and the
    /// policy and value heads.
    ///
    /// # Arguments
    ///
    /// * `version` - validated container version (1 or 2)
    /// * `architecture` - parsed header the tensor stream must implement
    ///
    /// # Errors
    ///
    /// Returns an error when any tensor dimension, count, activation, or
    /// residual-alpha value contradicts the architecture header.
    ///
    /// # Panics
    ///
    /// Panics if the v2 extensions are absent when the shared smolgen tensor
    /// is parsed; the preceding `is_some_and` guard makes this unreachable.
    fn parse_weights(
        &mut self,
        version: u32,
        architecture: &Bt4Architecture,
    ) -> Result<(), Bt4Error> {
        if version == 1 {
            let projected = match architecture.input_embedding {
                Bt4InputEmbedding::None => architecture.input_channels,
                Bt4InputEmbedding::PositionMap => checked_add(
                    architecture.input_channels,
                    architecture.tokens,
                    "position-map width",
                )?,
                Bt4InputEmbedding::PositionDense => {
                    return Err(shape("PE_DENSE is not part of BT4 v1"));
                }
            };
            let embedding = self.dense("input.embedding")?;
            expect_dense(
                embedding,
                projected,
                architecture.embedding_size,
                "input embedding",
            )?;
        } else {
            self.parse_v2_input(architecture)?;
        }
        #[allow(
            // The serialized model stores f32, while the Java exporter derives
            // this pinned constant in double precision before narrowing it.
            clippy::cast_possible_truncation
        )]
        let expected_residual_alpha = architecture
            .v2
            .as_ref()
            .map(|_| (2.0_f64 * f64::from(architecture.encoder_layers)).powf(-0.25) as f32);
        self.encoder_blocks(
            "body",
            architecture.encoder_layers,
            architecture.embedding_size,
            Some(architecture.attention_heads),
            architecture.v2.as_ref(),
            expected_residual_alpha,
        )?;
        if architecture
            .v2
            .as_ref()
            .is_some_and(|extensions| extensions.has_smolgen)
        {
            let extensions = architecture.v2.as_ref().expect("checked above");
            let expected = u64::from(checked_mul(
                extensions.smolgen_per_head_dim,
                extensions.smolgen_global_size,
                "shared smolgen tensor",
            )?);
            self.array("body.smolgen_w", Some(expected))?;
        }
        self.policy_head(architecture)?;
        self.value_head(architecture)?;
        Ok(())
    }

    /// Used for parsing optional v2 input preprocessing, gates, and embedding
    /// FFN tensors.
    ///
    /// The projected input width fed to the embedding dense layer depends on
    /// whether the preprocessing projection is present; with preprocessing,
    /// the preproc output must be divisible by the token count and widens the
    /// raw input channels accordingly.
    ///
    /// # Arguments
    ///
    /// * `architecture` - parsed version-2 header including its extensions
    ///
    /// # Errors
    ///
    /// Returns an error when a preprocessing, embedding, gate, or FFN tensor
    /// contradicts the widths declared by the architecture header.
    ///
    /// # Panics
    ///
    /// Panics if called for an architecture without v2 extensions;
    /// [`Self::parse_weights`] only invokes it for version-2 containers.
    fn parse_v2_input(&mut self, architecture: &Bt4Architecture) -> Result<(), Bt4Error> {
        let extension = architecture
            .v2
            .as_ref()
            .expect("v2 parser requires extensions");
        let projected = if extension.has_input_preproc {
            let preproc = self.dense("input.preproc")?;
            let expected_input = checked_mul(architecture.tokens, 12, "input preproc width")?;
            if preproc.input != expected_input {
                return Err(shape(format!(
                    "input preproc width is {}; expected tokens * 12 = {expected_input}",
                    preproc.input
                )));
            }
            if preproc.output % architecture.tokens != 0 {
                return Err(shape(
                    "input preproc output is not divisible by token count",
                ));
            }
            checked_add(
                architecture.input_channels,
                preproc.output / architecture.tokens,
                "preprocessed input width",
            )?
        } else {
            match architecture.input_embedding {
                Bt4InputEmbedding::None | Bt4InputEmbedding::PositionDense => {
                    architecture.input_channels
                }
                Bt4InputEmbedding::PositionMap => checked_add(
                    architecture.input_channels,
                    architecture.tokens,
                    "position-map width",
                )?,
            }
        };
        let embedding = self.dense("input.embedding")?;
        expect_dense(
            embedding,
            projected,
            architecture.embedding_size,
            "input embedding",
        )?;
        if architecture.input_embedding == Bt4InputEmbedding::PositionDense {
            self.array(
                "input.embedding_ln_gamma",
                Some(u64::from(architecture.embedding_size)),
            )?;
            self.array(
                "input.embedding_ln_beta",
                Some(u64::from(architecture.embedding_size)),
            )?;
        }
        if extension.has_input_gates {
            let gate = u64::from(checked_mul(
                architecture.tokens,
                architecture.embedding_size,
                "input gate",
            )?);
            self.array("input.mult_gate", Some(gate))?;
            self.array("input.add_gate", Some(gate))?;
        }
        if extension.has_input_embedding_ffn {
            let first = self.dense("input.ffn.in")?;
            expect_dense(
                first,
                architecture.embedding_size,
                extension.ffn_hidden_size,
                "input FFN first dense",
            )?;
            let second = self.dense("input.ffn.out")?;
            expect_dense(
                second,
                extension.ffn_hidden_size,
                architecture.embedding_size,
                "input FFN second dense",
            )?;
            self.array(
                "input.ffn_ln_gamma",
                Some(u64::from(architecture.embedding_size)),
            )?;
            self.array(
                "input.ffn_ln_beta",
                Some(u64::from(architecture.embedding_size)),
            )?;
        }
        Ok(())
    }

    /// Used for parsing an exact-size sequence of transformer encoder blocks.
    ///
    /// # Arguments
    ///
    /// * `prefix` - hierarchical name prefix recorded for block tensors
    /// * `expected_count` - block count the serialized sequence must match
    /// * `width` - token width shared by all blocks in the sequence
    /// * `expected_heads` - required head count, or `None` when each block
    ///   carries its own
    /// * `extension` - v2 extensions governing smolgen and FFN checks
    /// * `expected_residual_alpha` - pinned residual scale, when the
    ///   architecture defines one
    ///
    /// # Errors
    ///
    /// Returns an error when the serialized block count differs from
    /// `expected_count` or any contained block fails validation.
    fn encoder_blocks(
        &mut self,
        prefix: &str,
        expected_count: u32,
        width: u32,
        expected_heads: Option<u32>,
        extension: Option<&Bt4V2Extensions>,
        expected_residual_alpha: Option<f32>,
    ) -> Result<(), Bt4Error> {
        let count = self.count(&format!("{prefix} encoder blocks"), MAX_ENCODER_BLOCKS)?;
        if count != expected_count {
            return Err(shape(format!(
                "{prefix} encoder count {count} does not match architecture {expected_count}"
            )));
        }
        for block in 0..count {
            self.encoder_block(
                &format!("{prefix}.encoder[{block}]"),
                width,
                expected_heads,
                extension,
                expected_residual_alpha,
            )?;
        }
        Ok(())
    }

    /// Used for validating one attention/FFN block and its optional smolgen
    /// payload.
    ///
    /// Checks head divisibility, the four square attention projections, the
    /// smolgen compress/dense/normalization chain when enabled, the FFN pair,
    /// the four layer-norm vectors, the block activation against the v2
    /// default, and the positive residual-alpha constant.
    ///
    /// # Arguments
    ///
    /// * `prefix` - hierarchical name prefix recorded for block tensors
    /// * `width` - token width of the block input and output
    /// * `expected_heads` - required head count, or `None` when the block
    ///   carries its own
    /// * `extension` - v2 extensions governing smolgen and FFN checks
    /// * `expected_residual_alpha` - pinned residual scale, when the
    ///   architecture defines one
    ///
    /// # Errors
    ///
    /// Returns an error when any tensor, activation, or residual-alpha value
    /// in the block contradicts the expected shapes.
    fn encoder_block(
        &mut self,
        prefix: &str,
        width: u32,
        expected_heads: Option<u32>,
        extension: Option<&Bt4V2Extensions>,
        expected_residual_alpha: Option<f32>,
    ) -> Result<(), Bt4Error> {
        let heads = self.dimension(&format!("{prefix} attention heads"))?;
        if let Some(expected_heads) = expected_heads {
            if heads != expected_heads {
                return Err(shape(format!(
                    "{prefix} has {heads} attention heads; expected {expected_heads}"
                )));
            }
        }
        if width % heads != 0 {
            return Err(shape(format!(
                "{prefix} width {width} is not divisible by {heads} attention heads"
            )));
        }
        for name in ["query", "key", "value", "out"] {
            let dense = self.dense(&format!("{prefix}.attention.{name}"))?;
            expect_dense(dense, width, width, &format!("{prefix} attention {name}"))?;
        }
        if let Some(extension) = extension.filter(|extension| extension.has_smolgen) {
            let compress = self.dense(&format!("{prefix}.smolgen.compress"))?;
            expect_dense(
                compress,
                width,
                extension.smolgen_hidden_channels,
                &format!("{prefix} smolgen compress"),
            )?;
            let flattened = checked_mul(
                extension.smolgen_hidden_channels,
                // Smolgen always consumes every board token.
                exact_square_root(extension.smolgen_global_size)?,
                "smolgen flattened width",
            )?;
            let dense1 = self.dense(&format!("{prefix}.smolgen.dense1"))?;
            expect_dense(
                dense1,
                flattened,
                extension.smolgen_hidden_size,
                &format!("{prefix} smolgen dense1"),
            )?;
            self.array(
                &format!("{prefix}.smolgen.ln1_gamma"),
                Some(u64::from(extension.smolgen_hidden_size)),
            )?;
            self.array(
                &format!("{prefix}.smolgen.ln1_beta"),
                Some(u64::from(extension.smolgen_hidden_size)),
            )?;
            let dense2 = self.dense(&format!("{prefix}.smolgen.dense2"))?;
            let final_width =
                checked_mul(heads, extension.smolgen_per_head_dim, "smolgen final width")?;
            expect_dense(
                dense2,
                extension.smolgen_hidden_size,
                final_width,
                &format!("{prefix} smolgen dense2"),
            )?;
            self.array(
                &format!("{prefix}.smolgen.ln2_gamma"),
                Some(u64::from(final_width)),
            )?;
            self.array(
                &format!("{prefix}.smolgen.ln2_beta"),
                Some(u64::from(final_width)),
            )?;
        }
        let ffn_in = self.dense(&format!("{prefix}.ffn.in"))?;
        if ffn_in.input != width {
            return Err(shape(format!("{prefix} FFN input width mismatch")));
        }
        if let Some(extension) = extension {
            if ffn_in.output != extension.ffn_hidden_size {
                return Err(shape(format!("{prefix} FFN hidden width mismatch")));
            }
        }
        let ffn_out = self.dense(&format!("{prefix}.ffn.out"))?;
        expect_dense(
            ffn_out,
            ffn_in.output,
            width,
            &format!("{prefix} FFN output"),
        )?;
        for name in ["ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"] {
            self.array(&format!("{prefix}.{name}"), Some(u64::from(width)))?;
        }
        let activation = self.activation(&format!("{prefix} activation"))?;
        if extension.is_some_and(|extension| activation != extension.default_activation) {
            return Err(shape(format!(
                "{prefix} activation does not match the v2 default"
            )));
        }
        let alpha = self.f32(&format!("{prefix} residual alpha"))?;
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(shape(format!("{prefix} residual alpha must be positive")));
        }
        if expected_residual_alpha.is_some_and(|expected| (alpha - expected).abs() > 1.0e-5) {
            return Err(shape(format!(
                "{prefix} residual alpha does not match the architecture"
            )));
        }
        Ok(())
    }

    /// Used for parsing the policy transformer and promotion tensors.
    ///
    /// Reads the policy embedding, a policy-only encoder sequence whose blocks
    /// carry their own head counts, the query/key projection pair, the
    /// four-row promotion weight tensor, and the policy activation.
    ///
    /// # Arguments
    ///
    /// * `architecture` - parsed header supplying widths and the v2 default
    ///   activation
    ///
    /// # Errors
    ///
    /// Returns an error when embedding, query/key, promotion, or activation
    /// values contradict the expected policy-head shapes.
    fn policy_head(&mut self, architecture: &Bt4Architecture) -> Result<(), Bt4Error> {
        let embedding = self.dense("policy.embedding")?;
        if embedding.input != architecture.embedding_size {
            return Err(shape("policy embedding input width mismatch"));
        }
        let count = self.count("policy encoder blocks", MAX_ENCODER_BLOCKS)?;
        for block in 0..count {
            self.encoder_block(
                &format!("policy.encoder[{block}]"),
                embedding.output,
                // Policy-only encoders carry their own head count.
                None,
                None,
                None,
            )?;
        }
        let query = self.dense("policy.query")?;
        let key = self.dense("policy.key")?;
        if query.input != embedding.output
            || key.input != embedding.output
            || query.output != key.output
        {
            return Err(shape("policy query/key dimensions mismatch"));
        }
        self.array(
            "policy.promotion_weights",
            Some(u64::from(checked_mul(
                4,
                query.output,
                "policy promotion weights",
            )?)),
        )?;
        let activation = self.activation("policy activation")?;
        if architecture
            .v2
            .as_ref()
            .is_some_and(|extension| activation != extension.default_activation)
        {
            return Err(shape("policy activation does not match the v2 default"));
        }
        Ok(())
    }

    /// Used for parsing the dense WDL value head and verifying its three
    /// outputs.
    ///
    /// The value embedding is flattened over all tokens before the first
    /// dense layer; the second dense layer must emit exactly the three
    /// win/draw/loss outputs.
    ///
    /// # Arguments
    ///
    /// * `architecture` - parsed header supplying widths and the v2 default
    ///   activation
    ///
    /// # Errors
    ///
    /// Returns an error when the embedding, dense widths, WDL output count, or
    /// activation contradict the expected value-head shapes.
    fn value_head(&mut self, architecture: &Bt4Architecture) -> Result<(), Bt4Error> {
        let embedding = self.dense("value.embedding")?;
        if embedding.input != architecture.embedding_size {
            return Err(shape("value embedding input width mismatch"));
        }
        let fc1 = self.dense("value.fc1")?;
        let flattened = checked_mul(
            architecture.tokens,
            embedding.output,
            "value flattened width",
        )?;
        if fc1.input != flattened {
            return Err(shape("value first dense input width mismatch"));
        }
        let fc2 = self.dense("value.fc2")?;
        expect_dense(fc2, fc1.output, 3, "value WDL output")?;
        let activation = self.activation("value activation")?;
        if architecture
            .v2
            .as_ref()
            .is_some_and(|extension| activation != extension.default_activation)
        {
            return Err(shape("value activation does not match the v2 default"));
        }
        Ok(())
    }

    /// Used for parsing a row-major dense weight matrix followed by its bias
    /// vector.
    ///
    /// Records two manifest entries, `{name}.weights` with `input * output`
    /// elements and `{name}.bias` with `output` elements.
    ///
    /// # Arguments
    ///
    /// * `name` - hierarchical tensor name prefix for the weight/bias pair
    ///
    /// # Returns
    ///
    /// Input/output dimensions of the parsed layer.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is out of range, the element product
    /// overflows its bound, or either tensor fails validation.
    fn dense(&mut self, name: &str) -> Result<DenseShape, Bt4Error> {
        let input = self.dimension(&format!("{name} input dimension"))?;
        let output = self.dimension(&format!("{name} output dimension"))?;
        let weights = u64::from(checked_mul(input, output, &format!("{name} weights"))?);
        self.array(&format!("{name}.weights"), Some(weights))?;
        self.array(&format!("{name}.bias"), Some(u64::from(output)))?;
        Ok(DenseShape { input, output })
    }

    /// Used for recording and skipping one length-prefixed `f32` tensor.
    ///
    /// The optional expected length enforces the surrounding layer contract;
    /// the general per-tensor resource bound is enforced in either case.
    /// The payload is seeked over, not read, and its in-bounds range is
    /// appended to the manifest with the running parameter total.
    ///
    /// # Arguments
    ///
    /// * `name` - hierarchical tensor name recorded in the manifest
    /// * `expected` - exact element count required by the surrounding layer,
    ///   or `None` when only the global bound applies
    ///
    /// # Errors
    ///
    /// Returns an error when the element count violates a bound or the
    /// expected length, the byte range overflows or leaves the file, the
    /// parameter total overflows, or the forward seek fails.
    fn array(&mut self, name: &str, expected: Option<u64>) -> Result<(), Bt4Error> {
        let elements = u64::from(self.u32(&format!("{name} length"))?);
        if elements > MAX_TENSOR_ELEMENTS {
            return Err(shape(format!(
                "{name} has {elements} elements; safety limit is {MAX_TENSOR_ELEMENTS}"
            )));
        }
        if let Some(expected) = expected {
            if elements != expected {
                return Err(shape(format!(
                    "{name} has {elements} elements; expected {expected}"
                )));
            }
        }
        let bytes = elements
            .checked_mul(4)
            .ok_or_else(|| shape(format!("{name} byte length overflows")))?;
        let end = self
            .position
            .checked_add(bytes)
            .ok_or_else(|| shape(format!("{name} file offset overflows")))?;
        if end > self.file_bytes {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("{name} extends beyond the BT4 file"),
            ));
        }
        let parameter_count = self
            .parameter_count
            .checked_add(elements)
            .ok_or_else(|| shape("BT4 parameter count overflows"))?;
        let mut owned_name = String::new();
        owned_name.try_reserve_exact(name.len()).map_err(|_| {
            Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                format!("cannot reserve the BT4 tensor name {name}"),
            )
        })?;
        owned_name.push_str(name);
        self.tensors.try_reserve(1).map_err(|_| {
            Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                "cannot extend the BT4 tensor manifest",
            )
        })?;
        self.tensors.push(Bt4TensorInfo {
            name: owned_name,
            elements,
            byte_offset: self.position,
        });
        self.parameter_count = parameter_count;
        self.input
            .seek(SeekFrom::Start(end))
            .map_err(|error| Bt4Error::io(format!("cannot skip {name}"), error))?;
        self.position = end;
        Ok(())
    }

    /// Used for reading one length-prefixed activation name from the accepted
    /// vocabulary.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// Parsed [`Bt4Activation`] variant.
    ///
    /// # Errors
    ///
    /// Returns an error when the string cannot be read or is not one of
    /// `NONE`, `RELU`, `MISH`, `SWISH`, or `TANH`.
    fn activation(&mut self, label: &str) -> Result<Bt4Activation, Bt4Error> {
        match self.string(label)?.as_str() {
            "NONE" => Ok(Bt4Activation::None),
            "RELU" => Ok(Bt4Activation::Relu),
            "MISH" => Ok(Bt4Activation::Mish),
            "SWISH" => Ok(Bt4Activation::Swish),
            "TANH" => Ok(Bt4Activation::Tanh),
            other => Err(shape(format!("unsupported {label} {other}"))),
        }
    }

    /// Used for reading a canonical zero-or-one boolean byte.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// `false` for byte 0 and `true` for byte 1.
    ///
    /// # Errors
    ///
    /// Returns an error when the read fails or any other byte value is found.
    fn boolean(&mut self, label: &str) -> Result<bool, Bt4Error> {
        let mut byte = [0_u8; 1];
        self.read_exact(&mut byte, label)?;
        match byte[0] {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("{label} has invalid boolean byte {value}"),
            )),
        }
    }

    /// Used for reading a positive architecture dimension within
    /// [`MAX_DIMENSION`].
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// Dimension value in `1..=MAX_DIMENSION`.
    ///
    /// # Errors
    ///
    /// Returns an error when the read fails or the value is zero or above the
    /// bound.
    fn dimension(&mut self, label: &str) -> Result<u32, Bt4Error> {
        let value = self.u32(label)?;
        if value == 0 || value > MAX_DIMENSION {
            return Err(shape(format!(
                "{label} {value} is outside 1..={MAX_DIMENSION}"
            )));
        }
        Ok(value)
    }

    /// Used for reading a possibly-zero count bounded by a caller-supplied
    /// maximum.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    /// * `maximum` - inclusive safety limit for the count
    ///
    /// # Returns
    ///
    /// Count value in `0..=maximum`.
    ///
    /// # Errors
    ///
    /// Returns an error when the read fails or the value exceeds `maximum`.
    fn count(&mut self, label: &str, maximum: u32) -> Result<u32, Bt4Error> {
        let value = self.u32(label)?;
        if value > maximum {
            return Err(shape(format!(
                "{label} {value} exceeds safety limit {maximum}"
            )));
        }
        Ok(value)
    }

    /// Used for reading a bounded length-prefixed UTF-8 string.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// Decoded string contents.
    ///
    /// # Errors
    ///
    /// Returns an error when the declared length exceeds
    /// [`MAX_STRING_BYTES`], its allocation cannot be reserved, the bytes
    /// cannot be read, or they are not valid UTF-8.
    fn string(&mut self, label: &str) -> Result<String, Bt4Error> {
        let length = self.u32(&format!("{label} length"))?;
        if length > MAX_STRING_BYTES {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("{label} is too long: {length} bytes"),
            ));
        }
        let length = usize::try_from(length).map_err(|_| {
            Bt4Error::new(
                Bt4ErrorKind::ResourceLimit,
                format!("{label} length does not fit addressable memory"),
            )
        })?;
        let mut bytes = zeroed_bytes(length, label)?;
        self.read_exact(&mut bytes, label)?;
        String::from_utf8(bytes)
            .map_err(|_| Bt4Error::new(Bt4ErrorKind::Format, format!("{label} is not valid UTF-8")))
    }

    /// Used for reading one little-endian IEEE-754 bit pattern.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// Float value reinterpreted from the raw 32-bit field.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying four-byte read fails.
    fn f32(&mut self, label: &str) -> Result<f32, Bt4Error> {
        Ok(f32::from_bits(self.u32(label)?))
    }

    /// Used for reading one little-endian unsigned 32-bit field.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// Decoded unsigned value.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying four-byte read fails.
    fn u32(&mut self, label: &str) -> Result<u32, Bt4Error> {
        let mut bytes = [0_u8; 4];
        self.read_exact(&mut bytes, label)?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Used for reading an in-bounds header field and advancing the
    /// authoritative offset.
    ///
    /// The end offset is bounds-checked against the captured file length
    /// before any bytes are read, so a truncated model produces a format
    /// error rather than a short read.
    ///
    /// # Arguments
    ///
    /// * `output` - destination buffer defining the read length
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Errors
    ///
    /// Returns an error when the offset arithmetic overflows, the field would
    /// extend past the file, or the read itself fails.
    fn read_exact(&mut self, output: &mut [u8], label: &str) -> Result<(), Bt4Error> {
        let end = self
            .position
            .checked_add(u64::try_from(output.len()).map_err(|_| {
                Bt4Error::new(Bt4ErrorKind::Format, "BT4 read length does not fit u64")
            })?)
            .ok_or_else(|| Bt4Error::new(Bt4ErrorKind::Format, "BT4 offset overflow"))?;
        if end > self.file_bytes {
            return Err(Bt4Error::new(
                Bt4ErrorKind::Format,
                format!("truncated BT4 model while reading {label}"),
            ));
        }
        self.input
            .read_exact(output)
            .map_err(|error| Bt4Error::io(format!("cannot read {label} from BT4 model"), error))?;
        self.position = end;
        Ok(())
    }
}

/// Used for adding dimensions while enforcing the global supported-width
/// bound.
///
/// # Arguments
///
/// * `left` - first dimension operand
/// * `right` - second dimension operand
/// * `label` - diagnostic name for the combined width
///
/// # Returns
///
/// Sum of both operands.
///
/// # Errors
///
/// Returns an error when the sum overflows or exceeds [`MAX_DIMENSION`].
fn checked_add(left: u32, right: u32, label: &str) -> Result<u32, Bt4Error> {
    left.checked_add(right)
        .filter(|value| *value <= MAX_DIMENSION)
        .ok_or_else(|| shape(format!("{label} exceeds supported dimensions")))
}

/// Used for multiplying tensor dimensions while enforcing the element-count
/// bound.
///
/// # Arguments
///
/// * `left` - first dimension operand
/// * `right` - second dimension operand
/// * `label` - diagnostic name for the resulting element count
///
/// # Returns
///
/// Product of both operands.
///
/// # Errors
///
/// Returns an error when the product overflows or exceeds
/// [`MAX_TENSOR_ELEMENTS`].
fn checked_mul(left: u32, right: u32, label: &str) -> Result<u32, Bt4Error> {
    left.checked_mul(right)
        .filter(|value| u64::from(*value) <= MAX_TENSOR_ELEMENTS)
        .ok_or_else(|| shape(format!("{label} exceeds supported dimensions")))
}

/// Used for finding the integral square root required by a square smolgen
/// map.
///
/// # Arguments
///
/// * `value` - smolgen global size that must be a perfect square
///
/// # Returns
///
/// Exact square root of `value`.
///
/// # Errors
///
/// Returns an error when `value` is not a perfect square.
fn exact_square_root(value: u32) -> Result<u32, Bt4Error> {
    let mut root = 1_u32;
    while root.saturating_mul(root) < value {
        root = root.saturating_add(1);
    }
    if root.saturating_mul(root) != value {
        return Err(shape("smolgen global size is not a perfect square"));
    }
    Ok(root)
}

/// Used for checking one parsed dense layer against its architectural
/// input/output widths.
///
/// # Arguments
///
/// * `actual` - dimensions read from the serialized layer
/// * `input` - required input width
/// * `output` - required output width
/// * `label` - diagnostic name of the layer being checked
///
/// # Errors
///
/// Returns an error when either dimension differs from the expectation.
fn expect_dense(actual: DenseShape, input: u32, output: u32, label: &str) -> Result<(), Bt4Error> {
    if actual.input != input || actual.output != output {
        return Err(shape(format!(
            "{label} is {}x{}; expected {input}x{output}",
            actual.input, actual.output
        )));
    }
    Ok(())
}

/// Used for constructing a consistent unsupported-shape error.
///
/// # Arguments
///
/// * `message` - human-readable description of the shape violation
///
/// # Returns
///
/// Error of kind [`Bt4ErrorKind::UnsupportedShape`] carrying the message.
fn shape(message: impl Into<String>) -> Bt4Error {
    Bt4Error::new(Bt4ErrorKind::UnsupportedShape, message)
}

