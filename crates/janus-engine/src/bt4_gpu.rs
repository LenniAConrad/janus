//! Safe client for the persistent BT4 OpenCL helper process.
//!
//! Janus forbids unsafe Rust and deliberately has no third-party Rust
//! dependencies. GPU runtimes cannot be called directly under that contract, so
//! the native OpenCL implementation lives in a fault-isolated helper process.
//! This module owns the bounded, versioned binary protocol used over the helper's
//! standard input and output. The helper is long lived: model loading and device
//! initialization happen once, while every prediction transfers exactly 7,168
//! input floats and receives 4,288 policy logits plus three WDL probabilities.
//!
//! Frames and floats use little-endian byte order. No helper-provided length is
//! trusted before it is checked against the `MAX_FRAME_PAYLOAD_BYTES` cap. The protocol
//! is intentionally batch-size one because Janus's current BT4 MCTS search is
//! serial; a later batching protocol must use a new protocol version.
//!
//! The helper is invoked without a shell as
//! `janus-bt4-opencl --worker --protocol 1 --model PATH --vendor V --device N`.
//! Every frame begins with this 24-byte header:
//!
//! | Offset | Field |
//! | --- | --- |
//! | `0..4` | ASCII magic `JGPU` |
//! | `4..6` | protocol version `u16` |
//! | `6..8` | opcode `u16`; responses set bit `0x8000` |
//! | `8..16` | paired request identifier `u64` |
//! | `16..20` | payload byte length `u32` |
//! | `20..24` | response status `u32`; requests use zero |
//!
//! `HELLO` (opcode 1) has an empty request. Its success payload is six `u32`
//! fields—vendor (`1=NVIDIA`, `2=AMD`, `3=Intel`), device index, input float
//! count, policy float count, WDL float count, and device-name byte length—then
//! the UTF-8 device name. `PREDICT` (opcode 2) carries 7,168 `f32` values and
//! returns 4,288 policy `f32` values followed by three WDL `f32` values.
//! `SHUTDOWN` is opcode 3. A nonzero response status carries a bounded UTF-8
//! diagnostic instead of an operation-specific success payload.

#[cfg(feature = "gpu")]
use std::env;
use std::fmt;
#[cfg(feature = "gpu")]
use std::io::{self, Read, Write};
#[cfg(feature = "gpu")]
use std::mem::size_of;
#[cfg(feature = "gpu")]
use std::path::{Path, PathBuf};
#[cfg(feature = "gpu")]
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
#[cfg(feature = "gpu")]
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "gpu")]
use std::thread::{self, JoinHandle};
#[cfg(feature = "gpu")]
use std::time::Duration;

/// Used for sizing the encoded BT4 input of one protocol prediction.
///
/// One prediction transfers 112 planes of 64 squares, 7,168 floats.
pub const INPUT_FLOATS: usize = 112 * 64;
/// Used for sizing the internal BT4 policy logits returned by one
/// prediction.
///
/// One prediction returns 67 planes of 64 squares, 4,288 floats.
pub const POLICY_FLOATS: usize = 67 * 64;
/// Used for sizing the win/draw/loss probabilities returned by one
/// prediction.
pub const WDL_FLOATS: usize = 3;

/// Used for marking the beginning of every helper protocol frame with the
/// four ASCII bytes `JGPU`.
#[cfg(feature = "gpu")]
const FRAME_MAGIC: [u8; 4] = *b"JGPU";
/// Used for identifying the current incompatible protocol generation.
#[cfg(feature = "gpu")]
const PROTOCOL_VERSION: u16 = 1;
/// Used for sizing the fixed 24-byte frame header.
#[cfg(feature = "gpu")]
const FRAME_HEADER_BYTES: usize = 24;
/// Used for bounding the largest payload accepted from the helper.
///
/// The fixed prediction messages need less than 32 KiB. Keeping a 64 KiB cap
/// leaves room for bounded device metadata and diagnostics without permitting an
/// untrusted helper to request an arbitrary allocation.
#[cfg(feature = "gpu")]
const MAX_FRAME_PAYLOAD_BYTES: usize = 64 * 1024;
/// Used for bounding the UTF-8 device name retained from the helper
/// handshake.
#[cfg(feature = "gpu")]
const MAX_DEVICE_NAME_BYTES: usize = 256;
/// Used for bounding the UTF-8 diagnostic retained from a failed helper
/// response.
#[cfg(feature = "gpu")]
const MAX_ERROR_BYTES: usize = 4 * 1024;
/// Used for bounding process startup, `OpenCL` initialization, and model
/// upload time.
#[cfg(feature = "gpu")]
const HELLO_TIMEOUT: Duration = Duration::from_secs(180);
/// Used for bounding the response wait of one batch-size-one GPU prediction.
///
/// A backend slower than five seconds per leaf is not viable for Janus search;
/// bounding it here limits stop/clock overrun before the safe CPU retry.
#[cfg(feature = "gpu")]
const PREDICT_TIMEOUT: Duration = Duration::from_secs(5);
/// Used for identifying the empty startup handshake request (opcode 1).
#[cfg(feature = "gpu")]
const OPCODE_HELLO: u16 = 1;
/// Used for identifying the fixed-shape BT4 inference request (opcode 2).
#[cfg(feature = "gpu")]
const OPCODE_PREDICT: u16 = 2;
/// Used for identifying the best-effort process teardown request (opcode 3).
#[cfg(feature = "gpu")]
const OPCODE_SHUTDOWN: u16 = 3;
/// Used for marking every response opcode returned by the helper.
#[cfg(feature = "gpu")]
const RESPONSE_BIT: u16 = 0x8000;
/// Used for indicating a successful helper response status.
#[cfg(feature = "gpu")]
const STATUS_OK: u32 = 0;

/// BT4 execution backend requested by the engine or selected by
/// auto-detection.
///
/// `Auto` and `Cpu` are selection policies, while `Nvidia`, `Amd`, and
/// `Intel` name the concrete `OpenCL` vendors a helper handshake can confirm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bt4Backend {
    /// Used for trying the `OpenCL` helper while retaining the CPU
    /// implementation as a fallback.
    Auto,
    /// Used for running the safe scalar Rust implementation only.
    Cpu,
    /// Used for requiring an NVIDIA `OpenCL` device.
    Nvidia,
    /// Used for requiring an AMD `OpenCL` device.
    Amd,
    /// Used for requiring an Intel `OpenCL` device.
    Intel,
}

impl Bt4Backend {
    /// Used for retrieving the stable lower-case spelling passed to the
    /// native helper as its `--vendor` argument.
    ///
    /// # Returns
    ///
    /// Static backend name understood by the helper command line.
    #[cfg(feature = "gpu")]
    const fn helper_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Nvidia => "nvidia",
            Self::Amd => "amd",
            Self::Intel => "intel",
        }
    }

    /// Used for decoding one concrete vendor identifier from the helper
    /// handshake.
    ///
    /// # Arguments
    ///
    /// * `value` - wire vendor code where `1` is NVIDIA, `2` is AMD, and `3`
    ///   is Intel
    ///
    /// # Returns
    ///
    /// Matching concrete backend, or `None` for an unknown code.
    #[cfg(feature = "gpu")]
    fn from_wire(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Nvidia),
            2 => Some(Self::Amd),
            3 => Some(Self::Intel),
            _ => None,
        }
    }

    /// Used for checking whether a concrete helper selection satisfies this
    /// request.
    ///
    /// `Auto` accepts any concrete GPU vendor, `Cpu` accepts only itself, and
    /// an explicit vendor request accepts only that vendor.
    ///
    /// # Arguments
    ///
    /// * `active` - concrete backend reported by the helper handshake
    ///
    /// # Returns
    ///
    /// `true` when the handshake selection is admissible for this request.
    #[cfg(feature = "gpu")]
    fn accepts(self, active: Self) -> bool {
        match self {
            Self::Auto => matches!(active, Self::Nvidia | Self::Amd | Self::Intel),
            Self::Cpu => active == Self::Cpu,
            Self::Nvidia | Self::Amd | Self::Intel => self == active,
        }
    }
}

impl fmt::Display for Bt4Backend {
    /// Used for writing the stable user-facing backend name.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Returns
    ///
    /// Propagated formatter success or failure.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Auto => "Auto",
            Self::Cpu => "CPU",
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
        };
        formatter.write_str(name)
    }
}

/// Observable result of resolving and initializing a BT4 execution backend.
///
/// The status pairs the caller's requested selection policy with the concrete
/// backend currently serving predictions, and records the helper-reported
/// device identity or the reason an attempted GPU selection fell back to the
/// CPU.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bt4BackendStatus {
    /// Used for recording the user-requested selection policy.
    requested: Bt4Backend,
    /// Used for recording the backend currently used for predictions.
    active: Bt4Backend,
    /// Used for recording the requested or helper-confirmed device index.
    device_index: usize,
    /// Used for retaining the helper-reported concrete `OpenCL` device name.
    device_name: Option<String>,
    /// Used for retaining the reason an automatic GPU attempt fell back to
    /// the CPU.
    fallback_reason: Option<String>,
}

impl Bt4BackendStatus {
    /// Used for creating the status for a direct CPU selection.
    ///
    /// # Arguments
    ///
    /// * `device_index` - device index recorded for observability
    ///
    /// # Returns
    ///
    /// Status whose requested and active backends are both CPU.
    pub(crate) const fn cpu(device_index: usize) -> Self {
        Self {
            requested: Bt4Backend::Cpu,
            active: Bt4Backend::Cpu,
            device_index,
            device_name: None,
            fallback_reason: None,
        }
    }

    /// Used for creating the status for an automatic GPU failure followed by
    /// CPU fallback.
    ///
    /// # Arguments
    ///
    /// * `device_index` - device index that was attempted
    /// * `reason` - human-readable fallback explanation
    ///
    /// # Returns
    ///
    /// Status recording an `Auto` request now served by the CPU.
    pub(crate) fn fallback(device_index: usize, reason: impl Into<String>) -> Self {
        Self {
            requested: Bt4Backend::Auto,
            active: Bt4Backend::Cpu,
            device_index,
            device_name: None,
            fallback_reason: Some(reason.into()),
        }
    }

    /// Used for recording a runtime GPU failure that switched an admitted
    /// backend to CPU.
    ///
    /// # Arguments
    ///
    /// * `requested` - backend policy originally supplied by the caller
    /// * `device_index` - device index that had been active
    /// * `reason` - human-readable failure explanation
    ///
    /// # Returns
    ///
    /// Status recording the original request now served by the CPU.
    #[cfg(feature = "gpu")]
    pub(crate) fn runtime_fallback(
        requested: Bt4Backend,
        device_index: usize,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            requested,
            active: Bt4Backend::Cpu,
            device_index,
            device_name: None,
            fallback_reason: Some(reason.into()),
        }
    }

    /// Used for retrieving the backend policy supplied by the caller.
    #[must_use]
    pub const fn requested(&self) -> Bt4Backend {
        self.requested
    }

    /// Used for retrieving the concrete backend currently serving
    /// predictions.
    #[must_use]
    pub const fn active(&self) -> Bt4Backend {
        self.active
    }

    /// Used for retrieving the selected device's zero-based vendor-local
    /// index.
    #[must_use]
    pub const fn device_index(&self) -> usize {
        self.device_index
    }

    /// Used for retrieving the native device name when a GPU helper is
    /// active.
    #[must_use]
    pub fn device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    /// Used for retrieving why a requested GPU switched to CPU, if fallback
    /// occurred.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }
}

/// Context-rich helper startup, protocol, or execution failure.
///
/// The wrapped string carries one bounded human-readable diagnostic that
/// callers surface directly or convert into the engine's public error family.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub(crate) struct Bt4GpuError(String);

#[cfg(feature = "gpu")]
impl Bt4GpuError {
    /// Used for creating one helper failure diagnostic.
    ///
    /// # Arguments
    ///
    /// * `message` - complete human-readable failure reason
    ///
    /// # Returns
    ///
    /// Error wrapping the supplied message.
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// Used for adding an operation label to an I/O failure.
    ///
    /// # Arguments
    ///
    /// * `operation` - short protocol operation name, such as `write`
    /// * `error` - underlying standard-library I/O error
    ///
    /// # Returns
    ///
    /// Error naming the failed helper operation and its cause.
    fn io(operation: &str, error: &io::Error) -> Self {
        Self(format!("BT4 GPU helper {operation} failed: {error}"))
    }
}

#[cfg(feature = "gpu")]
impl fmt::Display for Bt4GpuError {
    /// Used for writing the bounded human-readable failure reason.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Returns
    ///
    /// Propagated formatter success or failure.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for Bt4GpuError {}

/// One decoded helper protocol frame.
///
/// Frames are produced by [`read_frame`] only after the magic, protocol
/// version, and payload length were validated.
#[cfg(feature = "gpu")]
#[derive(Debug)]
struct Frame {
    /// Used for carrying the request opcode or a response opcode marked with
    /// [`RESPONSE_BIT`].
    opcode: u16,
    /// Used for pairing this frame with one monotonic client request.
    request_id: u64,
    /// Used for reporting zero on success or a stable native helper error
    /// code.
    status: u32,
    /// Used for holding the already length-checked payload bytes.
    payload: Vec<u8>,
}

/// Bounded copy of native diagnostics drained independently from the
/// protocol.
///
/// The drain thread appends at most [`MAX_ERROR_BYTES`] bytes and records
/// whether further output was discarded.
#[cfg(feature = "gpu")]
#[derive(Debug, Default)]
struct HelperStderr {
    /// Used for retaining the first diagnostic bytes, capped by
    /// [`MAX_ERROR_BYTES`].
    bytes: Vec<u8>,
    /// Used for indicating that additional bytes were drained but
    /// deliberately not retained.
    truncated: bool,
}

/// Persistent single-model `OpenCL` worker connection.
///
/// Exactly one request may be in flight. This matches mutable [`super::Bt4Network`]
/// ownership and the current serial MCTS tree. Standard-output parsing happens on
/// a dedicated safe Rust thread so inference has a bounded timeout even though
/// standard-library pipe reads do not expose a platform-neutral timeout API.
#[cfg(feature = "gpu")]
pub(crate) struct Bt4GpuClient {
    /// Used for owning the native helper process and its `OpenCL` device
    /// context.
    child: Option<Child>,
    /// Used for writing the binary request stream owned by the search thread.
    input: ChildStdin,
    /// Used for receiving frames decoded by the bounded output-reader thread.
    responses: Receiver<Result<Frame, Bt4GpuError>>,
    /// Used for joining the output-reader thread during teardown.
    reader: Option<JoinHandle<()>>,
    /// Used for sharing the bounded native standard-error bytes with the
    /// drain thread.
    stderr: Arc<Mutex<HelperStderr>>,
    /// Used for joining the drain thread that prevents verbose drivers from
    /// blocking the helper.
    stderr_reader: Option<JoinHandle<()>>,
    /// Used for issuing the next nonzero request identifier.
    next_request_id: u64,
    /// Used for reporting the concrete backend and device confirmed by the
    /// startup handshake.
    status: Bt4BackendStatus,
}

#[cfg(feature = "gpu")]
impl fmt::Debug for Bt4GpuClient {
    /// Used for omitting operating-system handle internals while retaining
    /// useful identity.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Returns
    ///
    /// Propagated formatter success or failure.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Bt4GpuClient")
            .field(
                "process_id",
                &self.child.as_ref().map(std::process::Child::id),
            )
            .field("next_request_id", &self.next_request_id)
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu")]
impl Bt4GpuClient {
    /// Used for starting the configured helper, completing its versioned
    /// handshake, and verifying its fixed BT4 tensor contract.
    ///
    /// The helper path is resolved from `JANUS_BT4_OPENCL_HELPER` when set;
    /// otherwise it must sit beside the Janus executable as `janus-bt4-opencl`
    /// (`janus-bt4-opencl.exe` on Windows). No shell or ambient `PATH` lookup
    /// is used.
    ///
    /// # Arguments
    ///
    /// * `model_path` - BT4 model file passed to the helper for upload
    /// * `requested` - GPU backend policy; must not be [`Bt4Backend::Cpu`]
    /// * `device_index` - zero-based vendor-local device index
    ///
    /// # Returns
    ///
    /// Connected client whose handshake confirmed the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4GpuError`] when CPU execution is requested, the helper
    /// cannot be resolved or started, or the handshake fails validation.
    pub(crate) fn connect(
        model_path: &Path,
        requested: Bt4Backend,
        device_index: usize,
    ) -> Result<Self, Bt4GpuError> {
        if requested == Bt4Backend::Cpu {
            return Err(Bt4GpuError::new(
                "the CPU backend does not use an OpenCL helper",
            ));
        }
        let helper_path = helper_path()?;
        Self::connect_to(&helper_path, model_path, requested, device_index)
    }

    /// Used for starting one explicitly located helper; split out for
    /// deterministic tests and development builds that have not installed
    /// sibling executables yet.
    ///
    /// # Arguments
    ///
    /// * `helper_path` - exact helper executable to spawn without a shell
    /// * `model_path` - BT4 model file passed to the helper for upload
    /// * `requested` - GPU backend policy validated against the handshake
    /// * `device_index` - zero-based vendor-local device index
    ///
    /// # Returns
    ///
    /// Connected client with both reader threads running and the HELLO
    /// handshake completed.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4GpuError`] when the process cannot be spawned, a standard
    /// stream is unavailable, or the HELLO exchange fails validation.
    fn connect_to(
        helper_path: &Path,
        model_path: &Path,
        requested: Bt4Backend,
        device_index: usize,
    ) -> Result<Self, Bt4GpuError> {
        let mut child = Command::new(helper_path)
            .arg("--worker")
            .arg("--protocol")
            .arg(PROTOCOL_VERSION.to_string())
            .arg("--model")
            .arg(model_path)
            .arg("--vendor")
            .arg(requested.helper_name())
            .arg("--device")
            .arg(device_index.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                Bt4GpuError::new(format!(
                    "cannot start BT4 OpenCL helper {}: {error}",
                    helper_path.display()
                ))
            })?;
        let Some(input) = child.stdin.take() else {
            terminate_child_bounded(child);
            return Err(Bt4GpuError::new(
                "BT4 OpenCL helper has no writable standard input",
            ));
        };
        let Some(output) = child.stdout.take() else {
            terminate_child_bounded(child);
            return Err(Bt4GpuError::new(
                "BT4 OpenCL helper has no readable standard output",
            ));
        };
        let Some(stderr) = child.stderr.take() else {
            terminate_child_bounded(child);
            return Err(Bt4GpuError::new(
                "BT4 OpenCL helper has no readable standard error",
            ));
        };
        let (sender, responses) = mpsc::sync_channel(2);
        let reader = match spawn_helper_thread("janus-bt4-gpu-output", move || {
            let mut output = output;
            loop {
                match read_frame(&mut output) {
                    Ok(frame) => {
                        if sender.send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        return;
                    }
                }
            }
        }) {
            Ok(reader) => reader,
            Err(error) => {
                terminate_child_bounded(child);
                return Err(Bt4GpuError::new(format!(
                    "cannot start BT4 OpenCL response reader: {error}"
                )));
            }
        };
        let stderr_bytes = Arc::new(Mutex::new(HelperStderr::default()));
        let stderr_target = Arc::clone(&stderr_bytes);
        let stderr_reader = match spawn_helper_thread("janus-bt4-gpu-stderr", move || {
            drain_stderr(stderr, &stderr_target);
        }) {
            Ok(stderr_reader) => stderr_reader,
            Err(error) => {
                terminate_child_bounded(child);
                if reader.is_finished() {
                    let _ = reader.join();
                }
                return Err(Bt4GpuError::new(format!(
                    "cannot start BT4 OpenCL stderr reader: {error}"
                )));
            }
        };
        let mut client = Self {
            child: Some(child),
            input,
            responses,
            reader: Some(reader),
            stderr: stderr_bytes,
            stderr_reader: Some(stderr_reader),
            next_request_id: 1,
            status: Bt4BackendStatus::cpu(device_index),
        };
        let hello = client.transact(OPCODE_HELLO, &[], HELLO_TIMEOUT)?;
        client.status = decode_hello(&hello.payload, requested)?;
        Ok(client)
    }

    /// Used for retrieving the concrete helper selection established by the
    /// handshake.
    pub(crate) const fn status(&self) -> &Bt4BackendStatus {
        &self.status
    }

    /// Used for running one fixed-shape inference and overwriting `policy`
    /// with internal logits.
    ///
    /// # Arguments
    ///
    /// * `encoded` - exactly [`INPUT_FLOATS`] finite input floats
    /// * `policy` - destination for exactly [`POLICY_FLOATS`] logits
    ///
    /// # Returns
    ///
    /// Validated win/draw/loss probabilities for the encoded position.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4GpuError`] for wrong input or policy lengths, non-finite
    /// input, request allocation failure, or any transport, timeout, or
    /// response-validation failure.
    pub(crate) fn predict(
        &mut self,
        encoded: &[f32],
        policy: &mut [f32],
    ) -> Result<[f32; WDL_FLOATS], Bt4GpuError> {
        if encoded.len() != INPUT_FLOATS {
            return Err(Bt4GpuError::new(format!(
                "BT4 GPU input has {} floats; expected {INPUT_FLOATS}",
                encoded.len()
            )));
        }
        if policy.len() != POLICY_FLOATS {
            return Err(Bt4GpuError::new(format!(
                "BT4 GPU policy destination has {} floats; expected {POLICY_FLOATS}",
                policy.len()
            )));
        }
        if encoded.iter().any(|value| !value.is_finite()) {
            return Err(Bt4GpuError::new("BT4 GPU input is not finite"));
        }

        let mut payload = Vec::new();
        payload
            .try_reserve_exact(INPUT_FLOATS * size_of::<f32>())
            .map_err(|_| Bt4GpuError::new("cannot allocate BT4 GPU request"))?;
        for value in encoded {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let response = self.transact(OPCODE_PREDICT, &payload, PREDICT_TIMEOUT)?;
        decode_prediction(&response.payload, policy)
    }

    /// Used for writing one request and waiting for its paired response
    /// frame.
    ///
    /// # Arguments
    ///
    /// * `opcode` - request opcode without [`RESPONSE_BIT`]
    /// * `payload` - already-bounded request payload bytes
    /// * `timeout` - watchdog applied to the response wait
    ///
    /// # Returns
    ///
    /// Successful response frame whose identifier and opcode match the
    /// request.
    ///
    /// # Errors
    ///
    /// Returns [`Bt4GpuError`] on write or flush failure, response timeout, a
    /// closed response stream, a mismatched request identifier or opcode, or
    /// a nonzero helper status; protocol violations also terminate the
    /// helper.
    fn transact(
        &mut self,
        opcode: u16,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<Frame, Bt4GpuError> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        write_frame(&mut self.input, opcode, request_id, STATUS_OK, payload)?;
        self.input
            .flush()
            .map_err(|error| Bt4GpuError::io("flush", &error))?;
        let received = match self.responses.recv_timeout(timeout) {
            Ok(Ok(received)) => received,
            Ok(Err(error)) => return Err(self.terminate_with_diagnostic(error)),
            Err(RecvTimeoutError::Timeout) => {
                let error = Bt4GpuError::new(format!(
                    "BT4 GPU helper timed out after {} milliseconds",
                    timeout.as_millis()
                ));
                return Err(self.terminate_with_diagnostic(error));
            }
            Err(RecvTimeoutError::Disconnected) => {
                let error = Bt4GpuError::new("BT4 GPU helper closed its response stream");
                return Err(self.terminate_with_diagnostic(error));
            }
        };
        if received.request_id != request_id {
            self.terminate();
            return Err(Bt4GpuError::new(format!(
                "BT4 GPU response id {} does not match request {request_id}",
                received.request_id
            )));
        }
        if received.opcode != opcode | RESPONSE_BIT {
            self.terminate();
            return Err(Bt4GpuError::new(format!(
                "BT4 GPU response opcode {} does not match request {opcode}",
                received.opcode
            )));
        }
        if received.status != STATUS_OK {
            let diagnostic = decode_error(&received.payload);
            return Err(Bt4GpuError::new(format!(
                "BT4 GPU helper status {}: {diagnostic}",
                received.status
            )));
        }
        Ok(received)
    }

    /// Used for killing and reaping a helper whose protocol can no longer be
    /// trusted.
    ///
    /// After a short bounded wait, an unresponsive process is handed to a
    /// detached reaper thread so UCI teardown stays bounded even when a
    /// vendor driver blocks.
    fn terminate(&mut self) {
        let Some(child) = self.child.take() else {
            return;
        };
        terminate_child_bounded(child);
    }

    /// Used for terminating a failed helper and appending its bounded native
    /// diagnostic.
    ///
    /// # Arguments
    ///
    /// * `error` - primary protocol or transport failure
    ///
    /// # Returns
    ///
    /// The original error, extended with retained helper standard-error text
    /// when any is available.
    fn terminate_with_diagnostic(&mut self, error: Bt4GpuError) -> Bt4GpuError {
        self.terminate();
        self.join_finished_readers();
        let diagnostic = self.stderr_diagnostic();
        if diagnostic.is_empty() {
            error
        } else {
            Bt4GpuError::new(format!("{error}; helper stderr: {diagnostic}"))
        }
    }

    /// Used for joining reader threads only after they finished, avoiding
    /// descendant-pipe hangs.
    ///
    /// Threads still blocked on helper pipes are left running instead of
    /// joined, keeping teardown bounded.
    fn join_finished_readers(&mut self) {
        if self.reader.as_ref().is_some_and(JoinHandle::is_finished) {
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
        }
        if self
            .stderr_reader
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            if let Some(reader) = self.stderr_reader.take() {
                let _ = reader.join();
            }
        }
    }

    /// Used for building one control-character-safe copy of retained helper
    /// diagnostics.
    ///
    /// # Returns
    ///
    /// Retained standard-error text with control characters replaced by
    /// spaces, trailing spaces trimmed, and `...` appended when truncated.
    fn stderr_diagnostic(&self) -> String {
        let diagnostic = self
            .stderr
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut text = String::from_utf8_lossy(&diagnostic.bytes)
            .chars()
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .collect::<String>();
        while text.ends_with(' ') {
            text.pop();
        }
        if diagnostic.truncated {
            text.push_str("...");
        }
        text
    }
}

#[cfg(feature = "gpu")]
impl Drop for Bt4GpuClient {
    /// Used for sending a best-effort shutdown marker, then guaranteeing
    /// process reaping.
    fn drop(&mut self) {
        let request_id = self.next_request_id;
        let _ = write_frame(&mut self.input, OPCODE_SHUTDOWN, request_id, STATUS_OK, &[]);
        let _ = self.input.flush();
        self.terminate();
        self.join_finished_readers();
    }
}

/// Used for starting one named helper-reader thread without panicking when
/// the operating system refuses another thread.
///
/// # Arguments
///
/// * `name` - diagnostic thread name visible to host tooling
/// * `task` - reader or drain loop to execute
///
/// # Returns
///
/// Join handle for a started thread.
///
/// # Errors
///
/// Returns the operating-system spawn error without running `task`.
#[cfg(feature = "gpu")]
fn spawn_helper_thread<F>(name: &str, task: F) -> io::Result<JoinHandle<()>>
where
    F: FnOnce() + Send + 'static,
{
    thread::Builder::new().name(name.to_owned()).spawn(task)
}

/// Used for killing a failed native helper without allowing teardown to block
/// indefinitely or a reaper-thread refusal to panic Janus.
///
/// The child receives a kill request and is polled for at most about 100 ms.
/// A survivor is moved into a detached named reaper. If that final thread is
/// also refused, dropping its unrun task releases the already-killed child
/// handle; this can leave an OS zombie until process exit, but cannot unwind
/// the engine or block `bestmove`.
///
/// # Arguments
///
/// * `child` - helper process no longer trusted by the client
#[cfg(feature = "gpu")]
fn terminate_child_bounded(mut child: Child) {
    let _ = child.kill();
    for _ in 0..10 {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => thread::sleep(Duration::from_millis(10)),
        }
    }
    let _ = spawn_helper_thread("janus-bt4-gpu-reaper", move || {
        let _ = child.wait();
    });
}

/// Used for draining helper standard error forever while retaining only a
/// bounded prefix.
///
/// # Arguments
///
/// * `input` - helper standard-error pipe read until closure or failure
/// * `target` - shared bounded buffer receiving the retained prefix
#[cfg(feature = "gpu")]
fn drain_stderr(mut input: ChildStderr, target: &Mutex<HelperStderr>) {
    let mut chunk = [0_u8; 1_024];
    loop {
        let count = match input.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(count) => count,
        };
        let mut diagnostic = target
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let remaining = MAX_ERROR_BYTES.saturating_sub(diagnostic.bytes.len());
        let retained = remaining.min(count);
        diagnostic.bytes.extend_from_slice(&chunk[..retained]);
        if retained < count {
            diagnostic.truncated = true;
        }
    }
}

/// Used for resolving the helper without invoking a shell or searching
/// ambient `PATH`.
///
/// `JANUS_BT4_OPENCL_HELPER` overrides resolution when set; otherwise the
/// helper must sit beside the Janus executable under its platform-specific
/// name.
///
/// # Returns
///
/// Resolved helper executable path.
///
/// # Errors
///
/// Returns [`Bt4GpuError`] when the override variable is set but empty or the
/// Janus executable location cannot be determined.
#[cfg(feature = "gpu")]
fn helper_path() -> Result<PathBuf, Bt4GpuError> {
    if let Some(value) = env::var_os("JANUS_BT4_OPENCL_HELPER") {
        if value.is_empty() {
            return Err(Bt4GpuError::new("JANUS_BT4_OPENCL_HELPER is set but empty"));
        }
        return Ok(PathBuf::from(value));
    }
    let executable = env::current_exe()
        .map_err(|error| Bt4GpuError::new(format!("cannot resolve Janus executable: {error}")))?;
    let directory = executable.parent().ok_or_else(|| {
        Bt4GpuError::new("Janus executable has no parent directory for helper resolution")
    })?;
    let name = if cfg!(windows) {
        "janus-bt4-opencl.exe"
    } else {
        "janus-bt4-opencl"
    };
    Ok(directory.join(name))
}

/// Used for writing one already-bounded protocol frame.
///
/// # Arguments
///
/// * `output` - destination byte stream
/// * `opcode` - request or response opcode
/// * `request_id` - pairing identifier echoed by the matching response
/// * `status` - response status; requests use [`STATUS_OK`]
/// * `payload` - payload bytes bounded by [`MAX_FRAME_PAYLOAD_BYTES`]
///
/// # Errors
///
/// Returns [`Bt4GpuError`] for an oversized payload or a stream write
/// failure.
#[cfg(feature = "gpu")]
fn write_frame<W: Write>(
    output: &mut W,
    opcode: u16,
    request_id: u64,
    status: u32,
    payload: &[u8],
) -> Result<(), Bt4GpuError> {
    if payload.len() > MAX_FRAME_PAYLOAD_BYTES {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU payload has {} bytes; limit is {MAX_FRAME_PAYLOAD_BYTES}",
            payload.len()
        )));
    }
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| Bt4GpuError::new("BT4 GPU payload length does not fit u32"))?;
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    header[0..4].copy_from_slice(&FRAME_MAGIC);
    header[4..6].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    header[6..8].copy_from_slice(&opcode.to_le_bytes());
    header[8..16].copy_from_slice(&request_id.to_le_bytes());
    header[16..20].copy_from_slice(&payload_len.to_le_bytes());
    header[20..24].copy_from_slice(&status.to_le_bytes());
    output
        .write_all(&header)
        .and_then(|()| output.write_all(payload))
        .map_err(|error| Bt4GpuError::io("write", &error))
}

/// Used for reading and bounding one complete helper response frame.
///
/// # Arguments
///
/// * `input` - source byte stream positioned at a frame header
///
/// # Returns
///
/// Decoded frame whose payload length was checked against
/// [`MAX_FRAME_PAYLOAD_BYTES`] before allocation.
///
/// # Errors
///
/// Returns [`Bt4GpuError`] for read failures, an invalid magic, an
/// unsupported protocol version, an oversized payload, or allocation
/// failure.
#[cfg(feature = "gpu")]
fn read_frame<R: Read>(input: &mut R) -> Result<Frame, Bt4GpuError> {
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    input
        .read_exact(&mut header)
        .map_err(|error| Bt4GpuError::io("read header", &error))?;
    if header[0..4] != FRAME_MAGIC {
        return Err(Bt4GpuError::new("BT4 GPU frame has invalid magic"));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != PROTOCOL_VERSION {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU protocol version {version} is unsupported; expected {PROTOCOL_VERSION}"
        )));
    }
    let opcode = u16::from_le_bytes([header[6], header[7]]);
    let request_id = u64::from_le_bytes(
        header[8..16]
            .try_into()
            .expect("fixed frame request id has eight bytes"),
    );
    let payload_len = u32::from_le_bytes(
        header[16..20]
            .try_into()
            .expect("fixed frame payload length has four bytes"),
    ) as usize;
    let status = u32::from_le_bytes(
        header[20..24]
            .try_into()
            .expect("fixed frame status has four bytes"),
    );
    if payload_len > MAX_FRAME_PAYLOAD_BYTES {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU response payload has {payload_len} bytes; limit is {MAX_FRAME_PAYLOAD_BYTES}"
        )));
    }
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(payload_len)
        .map_err(|_| Bt4GpuError::new("cannot allocate BT4 GPU response"))?;
    payload.resize(payload_len, 0);
    input
        .read_exact(&mut payload)
        .map_err(|error| Bt4GpuError::io("read payload", &error))?;
    Ok(Frame {
        opcode,
        request_id,
        status,
        payload,
    })
}

/// Used for decoding the successful HELLO payload and validating its fixed
/// model shape.
///
/// # Arguments
///
/// * `payload` - successful HELLO response payload
/// * `requested` - backend policy the confirmed vendor must satisfy
///
/// # Returns
///
/// Backend status naming the confirmed vendor, device index, and device
/// name.
///
/// # Errors
///
/// Returns [`Bt4GpuError`] for a payload shorter than 24 bytes, a device
/// index that does not fit `usize`, a tensor-shape mismatch, an invalid
/// device-name length, an unknown vendor code, a vendor the request does not
/// accept, or a non-UTF-8 or empty device name.
#[cfg(feature = "gpu")]
fn decode_hello(payload: &[u8], requested: Bt4Backend) -> Result<Bt4BackendStatus, Bt4GpuError> {
    if payload.len() < 24 {
        return Err(Bt4GpuError::new(
            "BT4 GPU HELLO payload is shorter than 24 bytes",
        ));
    }
    let vendor = read_u32(payload, 0)?;
    let device_index = usize::try_from(read_u32(payload, 4)?)
        .map_err(|_| Bt4GpuError::new("BT4 GPU device index does not fit usize"))?;
    let input_floats = read_u32(payload, 8)? as usize;
    let policy_floats = read_u32(payload, 12)? as usize;
    let wdl_floats = read_u32(payload, 16)? as usize;
    let name_len = read_u32(payload, 20)? as usize;
    if input_floats != INPUT_FLOATS || policy_floats != POLICY_FLOATS || wdl_floats != WDL_FLOATS {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU helper shape is {input_floats}/{policy_floats}/{wdl_floats}; expected {INPUT_FLOATS}/{POLICY_FLOATS}/{WDL_FLOATS}"
        )));
    }
    if name_len > MAX_DEVICE_NAME_BYTES || payload.len() != 24 + name_len {
        return Err(Bt4GpuError::new(
            "BT4 GPU HELLO device-name length is invalid",
        ));
    }
    let active = Bt4Backend::from_wire(vendor).ok_or_else(|| {
        Bt4GpuError::new(format!("BT4 GPU HELLO has unknown vendor code {vendor}"))
    })?;
    if !requested.accepts(active) {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU helper selected {active}, but {requested} was requested"
        )));
    }
    let device_name = std::str::from_utf8(&payload[24..])
        .map_err(|_| Bt4GpuError::new("BT4 GPU device name is not UTF-8"))?;
    if device_name.is_empty() {
        return Err(Bt4GpuError::new("BT4 GPU device name is empty"));
    }
    Ok(Bt4BackendStatus {
        requested,
        active,
        device_index,
        device_name: Some(device_name.to_owned()),
        fallback_reason: None,
    })
}

/// Used for decoding fixed-size policy and WDL float arrays into
/// caller-owned storage.
///
/// # Arguments
///
/// * `payload` - successful PREDICT response payload
/// * `policy` - destination overwritten with [`POLICY_FLOATS`] logits
///
/// # Returns
///
/// Win/draw/loss probabilities, each inside `[0, 1]` and summing to
/// approximately one.
///
/// # Errors
///
/// Returns [`Bt4GpuError`] for a wrong payload size, non-finite outputs,
/// out-of-range WDL probabilities, or a WDL sum outside the tolerance.
#[cfg(feature = "gpu")]
fn decode_prediction(payload: &[u8], policy: &mut [f32]) -> Result<[f32; WDL_FLOATS], Bt4GpuError> {
    let expected = (POLICY_FLOATS + WDL_FLOATS) * size_of::<f32>();
    if payload.len() != expected {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU prediction has {} bytes; expected {expected}",
            payload.len()
        )));
    }
    for (index, output) in policy.iter_mut().enumerate() {
        *output = read_f32(payload, index * size_of::<f32>())?;
    }
    let mut wdl = [0.0_f32; WDL_FLOATS];
    let wdl_start = POLICY_FLOATS * size_of::<f32>();
    for (index, output) in wdl.iter_mut().enumerate() {
        *output = read_f32(payload, wdl_start + index * size_of::<f32>())?;
    }
    if policy.iter().any(|value| !value.is_finite()) || wdl.iter().any(|value| !value.is_finite()) {
        return Err(Bt4GpuError::new(
            "BT4 GPU prediction contains a non-finite value",
        ));
    }
    if wdl.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(Bt4GpuError::new(
            "BT4 GPU WDL probability is outside [0, 1]",
        ));
    }
    let wdl_sum: f32 = wdl.iter().sum();
    if (wdl_sum - 1.0).abs() > 1.0e-3 {
        return Err(Bt4GpuError::new(format!(
            "BT4 GPU WDL probabilities sum to {wdl_sum}; expected approximately 1"
        )));
    }
    Ok(wdl)
}

/// Used for reading one little-endian `u32` at an already-bounded byte
/// offset.
///
/// # Arguments
///
/// * `bytes` - source payload
/// * `offset` - byte offset of the value
///
/// # Returns
///
/// Decoded value.
///
/// # Errors
///
/// Returns [`Bt4GpuError`] when the payload ends before the value.
#[cfg(feature = "gpu")]
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Bt4GpuError> {
    let value = bytes
        .get(offset..offset + size_of::<u32>())
        .ok_or_else(|| Bt4GpuError::new("BT4 GPU payload is truncated"))?;
    Ok(u32::from_le_bytes(
        value
            .try_into()
            .expect("bounded u32 payload slice has four bytes"),
    ))
}

/// Used for reading one finite-or-nonfinite little-endian float for later
/// validation.
///
/// # Arguments
///
/// * `bytes` - source payload
/// * `offset` - byte offset of the value
///
/// # Returns
///
/// Bit-exact decoded float; finiteness is checked by the caller.
///
/// # Errors
///
/// Returns [`Bt4GpuError`] when the payload ends before the value.
#[cfg(feature = "gpu")]
fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, Bt4GpuError> {
    Ok(f32::from_bits(read_u32(bytes, offset)?))
}

/// Used for converting an error response payload to a bounded printable
/// diagnostic.
///
/// # Arguments
///
/// * `payload` - error response payload, possibly empty or non-UTF-8
///
/// # Returns
///
/// Lossy UTF-8 text capped at [`MAX_ERROR_BYTES`], with `...` appended when
/// truncated and a placeholder when empty.
#[cfg(feature = "gpu")]
fn decode_error(payload: &[u8]) -> String {
    let bounded = &payload[..payload.len().min(MAX_ERROR_BYTES)];
    let mut message = String::from_utf8_lossy(bounded).into_owned();
    if payload.len() > MAX_ERROR_BYTES {
        message.push_str("...");
    }
    if message.is_empty() {
        message.push_str("no diagnostic supplied");
    }
    message
}
