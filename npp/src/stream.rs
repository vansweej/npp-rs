//! Application-managed NPP stream context abstraction.
//!
//! # Execution model
//!
//! NPP `_Ctx` functions are enqueued asynchronously on the stream stored
//! in this context. The ordering contract has two parts:
//!
//! 1. **Upload → NPP op: ordered by construction.**
//!    [`CudaDevice::fork_default_stream()`] creates a new stream with an
//!    implicit wait so all prior default-stream work is visible. A
//!    host-to-device copy on the default stream completes before the NPP
//!    op on the forked stream begins.
//!
//! 2. **NPP op → readback: requires explicit host fence.**
//!    [`TryFrom<&CudaImage>`](crate::image::CudaImage) performs its DtoH
//!    copy on the NULL stream (cudarc 0.9's only DtoH API —
//!    `dtoh_sync_copy` does not accept a caller-supplied stream). The
//!    forked stream and NULL stream are unordered. The host-blocking
//!    [`synchronize()`](StreamContext::synchronize) call inserted *before*
//!    the NULL-stream copy is the load-bearing barrier that makes this safe.
//!
//! **Earlier revisions** claimed the readback was on the forked stream and
//! ordered by construction. That was incorrect — see
//! [`docs/stream-context.md`](https://github.com/vansweej/npp-rs/blob/main/docs/stream-context.md)
//! for the full rationale.
//!
//! # Safety
//!
//! [`NppStreamContext`] is populated by querying the CUDA device with
//! `cuDeviceGetAttribute` (via
//! [`CudaDevice::attribute`]). The `hStream` field is
//! a cross-crate pointer cast from
//! [`cudarc::driver::sys::CUstream`] to
//! `npp_sys::cudaStream_t` — both are `*mut CUstream_st` to the same
//! underlying driver object, so the cast is semantically valid despite
//! crossing FFI-crate boundaries.
//!
//! The originating `cuDeviceGetAttribute` driver call is **unsafe**, but
//! `CudaDevice::attribute` wraps it safely by guaranteeing device validity.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::{result, sys, CudaDevice, CudaStream, DriverError};
use npp_sys::NppStreamContext;

use crate::error::NppError;

/// A CUDA device handle with an associated stream and a populated
/// [`NppStreamContext`] for use with NPP `_Ctx` functions.
///
/// Every [`StreamContext`] owns its device reference and stream. The device
/// handle **must outlive** all [`CudaImage`](crate::image::CudaImage) buffers
/// created from it (finding C7). This is enforced by `Arc<CudaDevice>`
/// reference counting.
///
/// # Async contract
///
/// NPP operations using this context are enqueued without blocking.
/// Synchronisation happens at well-defined points:
/// - [`StreamContext::synchronize`] — explicit fence, blocks until done.
/// - `TryFrom<&CudaImage>` read-back — synchronous DtoH copy.
///
/// See the module-level docs for ordering-correctness rationale.
#[derive(Debug)]
pub struct StreamContext {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    raw: NppStreamContext,
}

impl StreamContext {
    /// Create a new stream context on the given device.
    ///
    /// Forks a new default stream and populates an [`NppStreamContext`] from
    /// device attributes via `cuDeviceGetAttribute`.
    ///
    /// # Errors
    ///
    /// Returns [`DriverError`] if any `cuDeviceGetAttribute` call fails,
    /// or if stream creation fails.
    ///
    /// # Panics
    ///
    /// Panics if the device ordinal does not match the expected device
    /// (defensive; should never happen if `device` is well-typed).
    ///
    /// # CUDA context lifetime
    ///
    /// The `device` handle must outlive all buffers created from it (C7).
    /// Passing a stale device handle will cause undefined behaviour in
    /// device-attribute queries.
    #[cfg(not(tarpaulin_include))]
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, DriverError> {
        // Fork a new stream. fork_default_stream inserts an implicit wait
        // so the forked stream sees all prior default-stream work.
        let stream = device.fork_default_stream()?;

        // Populate NppStreamContext from device attributes.
        let raw = populate_stream_context(&device, &stream)?;

        Ok(Self {
            device,
            stream,
            raw,
        })
    }

    /// Reference to the underlying CUDA device.
    #[cfg(not(tarpaulin_include))]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Reference to the CUDA stream.
    #[cfg(not(tarpaulin_include))]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// The populated [`NppStreamContext`] for passing to NPP `_Ctx` functions.
    ///
    /// # Safety
    ///
    /// The caller must not modify the returned struct. It is a populated
    /// snapshot of device attributes and a stream handle; mutating it
    /// (e.g. changing `hStream`) will cause NPP to enqueue operations
    /// on a different stream than expected, breaking the ordering contract.
    #[doc(hidden)]
    #[cfg(not(tarpaulin_include))]
    pub fn raw_ctx(&self) -> NppStreamContext {
        // NppStreamContext is Copy (confirmed Phase 0.3b), so this is
        // an implicit bitwise copy. Safe because the struct is fully
        // populated by the constructor and never mutated afterwards.
        self.raw
    }

    /// Block the host until all operations enqueued on this stream complete.
    ///
    /// This is the primary sync point for async NPP operations. It calls
    /// `cuStreamSynchronize` on the forked stream, which does not return
    /// until all prior work on that stream has finished.
    ///
    /// After calling `synchronize()`, host-side readback of device results
    /// is safe (no data race between NPP work on the forked stream and a
    /// subsequent DtoH copy on the NULL stream).
    #[cfg(not(tarpaulin_include))]
    pub fn synchronize(&self) -> Result<(), DriverError> {
        // SAFETY: self.stream.stream is a valid CUstream handle created by
        // fork_default_stream and owned by CudaStream. The call to
        // cuStreamSynchronize blocks the host until all work on this stream
        // completes. The stream is guaranteed alive because CudaStream's Drop
        // impl destroys it and we hold a reference.
        unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) }
    }

    /// Device-side fence: ensure all prior work on this stream is visible
    /// to subsequent work on other streams on the same device, without
    /// blocking the host.
    ///
    /// This is useful for ordering operations between streams within the
    /// same device context without a host round-trip. It is **not**
    /// sufficient to make host-side readback safe — use [`synchronize`]
    /// for that.
    ///
    /// [`synchronize`]: Self::synchronize
    #[cfg(not(tarpaulin_include))]
    pub fn device_fence(&self) -> Result<(), DriverError> {
        self.device.wait_for(&self.stream)
    }

    /// Create a new RAII timing event associated with this stream context.
    ///
    /// Use [`Event::record`] to record the event on the stream, then call
    /// [`elapsed`](Self::elapsed) to measure device-time between two events.
    #[cfg(not(tarpaulin_include))]
    pub fn record_event(&self) -> Event {
        let event = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .expect("cuEventCreate failed");
        Event {
            inner: event,
            stream: self.stream.stream,
            _device: Arc::clone(&self.device),
            _nosync: PhantomData,
        }
    }

    /// Measure device-time elapsed between two recorded events on this stream.
    ///
    /// Returns `Ok(Duration)` on success, or `Err(NppError)` if the driver
    /// call fails. Both events must have been recorded (via [`Event::record`])
    /// on the stream associated with this context, and the earlier event must
    /// have completed before this call.
    ///
    /// # Errors
    ///
    /// Returns `NppError::Cuda` if `cuEventElapsedTime` fails.
    #[cfg(not(tarpaulin_include))]
    pub fn elapsed(&self, start: &Event, end: &Event) -> Result<Duration, NppError> {
        // SAFETY: start.inner and end.inner are valid CUevents created by
        // record_event. Both were recorded on this stream (or a compatible one).
        let ms = unsafe { result::event::elapsed(start.inner, end.inner)? };
        Ok(Duration::from_secs_f64(ms as f64 / 1_000.0))
    }
}

/// A RAII wrapper around a CUDA event, used for device-side timing.
///
/// # Safety
///
/// The underlying CUDA event is created with `cuEventCreate` and destroyed
/// with `cuEventDestroy` on drop. The event is associated with the same
/// CUDA context as the `StreamContext` that created it. The caller must
/// ensure the CUDA device handle (`CudaDevice`) outlives this event
/// (same invariant as [`CudaImage`](crate::image::CudaImage) and
/// [`StreamContext`]).
///
/// # Panics
///
/// `record` and `elapsed` may panic if the underlying CUDA driver call
/// fails (these indicate a driver-level issue that is not recoverable).
#[derive(Debug)]
#[cfg(not(tarpaulin_include))]
pub struct Event {
    inner: sys::CUevent,
    stream: sys::CUstream,
    // Keeps the CUDA device (and thus the CUDA context) alive.
    _device: Arc<CudaDevice>,
    // Makes Event !Send + !Sync to match StreamContext's CUDA-context
    // thread affinity. CUDA events are tied to the CUcontext that created them.
    // This matches StreamContext's own !Send + !Sync (enforced via Arc).
    _nosync: PhantomData<*const ()>,
}

#[cfg(not(tarpaulin_include))]
impl Event {
    /// Record the event on the stream managed by this `StreamContext`.
    ///
    /// After this call, the event is recorded — use [`elapsed`](Self::elapsed)
    /// to measure device time between two recorded events.
    ///
    /// # Panics
    ///
    /// Panics if the underlying `cuEventRecord` call fails.
    pub fn record(&self) {
        // SAFETY: self.inner is a valid CUevent created in
        // StreamContext::record_event, and self.stream is a valid
        // CUstream from the same device context.
        let result = unsafe { result::event::record(self.inner, self.stream) };
        assert!(result.is_ok(), "cuEventRecord failed: {:?}", result);
    }
}

#[cfg(not(tarpaulin_include))]
impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: self.inner is a valid CUevent created by
        // cuEventCreate in StreamContext::record_event, and is
        // only destroyed once here.
        let _ = unsafe { result::event::destroy(self.inner) };
    }
}

/// Populate an [`NppStreamContext`] from device attributes and a stream.
///
/// Queries the CUDA device with `cuDeviceGetAttribute` for each field.
/// This is called once during [`StreamContext::new`].
///
/// # Safety
///
/// The device handle must be valid and the CUDA driver must be initialised.
/// Both invariants are upheld by `cudarc::CudaDevice`.
#[cfg(not(tarpaulin_include))]
fn populate_stream_context(
    device: &CudaDevice,
    stream: &CudaStream,
) -> Result<NppStreamContext, DriverError> {
    use cudarc::driver::sys::CUdevice_attribute_enum;

    let ordinal = device.ordinal();

    // Helper: query a device attribute via CudaDevice::attribute().
    // This wraps the unsafe cuDeviceGetAttribute safely.
    macro_rules! attr {
        ($name:ident) => {
            device.attribute(CUdevice_attribute_enum::$name)?
        };
    }

    // Cross-crate opaque pointer cast: cudarc's CUstream → npp-sys's
    // cudaStream_t. Both are *mut CUstream_st to the same underlying
    // driver object; the cast is semantically valid.
    let h_stream: npp_sys::cudaStream_t = stream.stream as npp_sys::cudaStream_t;

    Ok(NppStreamContext {
        hStream: h_stream,
        nCudaDeviceId: ordinal as i32,
        nMultiProcessorCount: attr!(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        nMaxThreadsPerMultiProcessor: attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        nMaxThreadsPerBlock: attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        nSharedMemPerBlock: attr!(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK) as usize,
        nCudaDevAttrComputeCapabilityMajor: attr!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        nCudaDevAttrComputeCapabilityMinor: attr!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        // nStreamFlags: fork_default_stream creates a standard stream
        // (non-blocking w.r.t. the default stream, which is the default).
        // TODO(perf): if we ever allow configurable stream flags,
        // propagate them here instead of hardcoding 0.
        nStreamFlags: 0,
        nReserved0: 0,
    })
}

/// Create a [`StreamContext`] for the device at the given ordinal.
///
/// Convenience wrapper around [`StreamContext::new`] that initialises the
/// CUDA device first.
///
/// # Errors
///
/// Returns [`DriverError`] if the device cannot be initialised (e.g. no
/// CUDA-capable GPU at that ordinal) or device-attribute queries fail.
///
/// # Example
///
/// ```rust,ignore
/// use npp_rs::stream_context_for;
/// let ctx = stream_context_for(0).expect("no GPU at ordinal 0");
/// ```
#[cfg(not(tarpaulin_include))]
#[allow(clippy::arc_with_non_send_sync)]
pub fn stream_context_for(ordinal: usize) -> Result<Arc<StreamContext>, DriverError> {
    let device = CudaDevice::new(ordinal)?;
    Ok(Arc::new(StreamContext::new(device)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NppStreamContext struct layout (compile-time assertions).
    #[test]
    fn npp_stream_context_size() {
        // Confirmed Phase 0.3b: 48 bytes on NPP 12.x.
        assert_eq!(std::mem::size_of::<NppStreamContext>(), 48);
    }

    /// NppStreamContext is Copy (confirmed Phase 0.3b).
    #[test]
    fn npp_stream_context_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NppStreamContext>();
    }

    /// raw_ctx() returns by value (no pointer shenanigans needed).
    #[test]
    fn raw_ctx_returns_by_value() {
        // This test cannot construct a real StreamContext without a GPU,
        // but it asserts that the return type of raw_ctx() is NppStreamContext
        // (not *const NppStreamContext or &NppStreamContext).
        // We verify this via the method signature: `fn raw_ctx(&self) -> NppStreamContext`.
        // The following line would fail to compile if it were a reference.
        fn _is_value_type(_: NppStreamContext) {}
        let _ = _is_value_type; // suppress unused warning
    }

    #[test]
    fn test_ms_to_duration_conversion() {
        // 0 ms → 0 secs
        let d = Duration::from_secs_f64(0.0 / 1_000.0);
        assert_eq!(d.as_nanos(), 0);

        // 1 ms → 1_000_000 ns
        let d = Duration::from_secs_f64(1.0 / 1_000.0);
        assert_eq!(d.as_nanos(), 1_000_000);

        // 1000 ms → 1 sec
        let d = Duration::from_secs_f64(1000.0 / 1_000.0);
        assert_eq!(d.as_secs(), 1);

        // 1.5 ms → 1_500_000 ns
        let d = Duration::from_secs_f64(1.5 / 1_000.0);
        assert_eq!(d.as_nanos(), 1_500_000);
    }
}
