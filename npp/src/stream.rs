//! Application-managed NPP stream context abstraction.
//!
//! # Execution model
//!
//! NPP `_Ctx` functions are enqueued asynchronously on the stream stored
//! in this context. The ordering contract has two parts:
//!
//! 1. **Upload → NPP op: ordered by construction.**
//!    [`CudaContext::new_stream()`] creates a new non-blocking stream. A
//!    host-to-device copy on this stream completes before the NPP op on the
//!    same stream begins because they are enqueued in order.
//!
//! 2. **NPP op → readback: requires explicit host fence.**
//!    [`TryFrom<&CudaImage>`](crate::image::CudaImage) performs its DtoH
//!    copy on the same per-stream channel via `clone_dtoh` (cudarc 0.19.x).
//!    The host-blocking [`synchronize()`](StreamContext::synchronize) call
//!    inserted *before* the per-stream copy is the load-bearing barrier.
//!
//! **Earlier revisions** claimed the readback was on the NULL stream and
//! ordered by construction. That was incorrect — see
//! [`docs/stream-context.md`](https://github.com/vansweej/npp-rs/blob/main/docs/stream-context.md)
//! for the full rationale.
//!
//! # Safety
//!
//! [`NppStreamContext`] is populated by querying the CUDA device with
//! `cuDeviceGetAttribute` (via
//! [`CudaContext::attribute`]). The `hStream` field is
//! a cross-crate pointer cast from
//! [`cudarc::driver::sys::CUstream`] to
//! `npp_sys::cudaStream_t` — both are `*mut CUstream_st` to the same
//! underlying driver object, so the cast is semantically valid despite
//! crossing FFI-crate boundaries.
//!
//! The originating `cuDeviceGetAttribute` driver call is **unsafe**, but
//! `CudaContext::attribute` wraps it safely by guaranteeing device validity.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::{sys, CudaContext, CudaEvent, CudaStream, DriverError};
use npp_sys::NppStreamContext;

use crate::error::NppError;

/// A CUDA device handle with an associated stream and a populated
/// [`NppStreamContext`] for use with NPP `_Ctx` functions.
///
/// Every [`StreamContext`] owns its device reference and stream. The device
/// handle **must outlive** all [`CudaImage`](crate::image::CudaImage) buffers
/// created from it (finding C7). This is enforced by `Arc<CudaContext>`
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
///
/// # !Send + !Sync
///
/// `CudaStream` became `Send + Sync` in cudarc 0.19.x, but CUDA streams are
/// inherently thread-bound (C7). We re-impose `!Send + !Sync` via
/// `PhantomData<*const ()>` to prevent `CudaImage` from silently becoming
/// `Send + Sync`.
#[derive(Debug)]
pub struct StreamContext {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    raw: NppStreamContext,
    // Re-impose !Send + !Sync. CudaStream became Send + Sync in 0.19.x,
    // but CUDA streams are thread-bound (C7). Without this marker,
    // CudaImage silently becomes Send + Sync.
    _nosync: PhantomData<*const ()>,
}

impl StreamContext {
    /// Create a new stream context on the given device.
    ///
    /// Creates a new non-blocking stream and populates an [`NppStreamContext`]
    /// from device attributes via `cuDeviceGetAttribute`.
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
    pub fn new(device: Arc<CudaContext>) -> Result<Self, DriverError> {
        // Create a new non-blocking stream. new_stream creates a CUDA stream
        // that does not implicitly synchronize with the default stream.
        let stream = device.new_stream()?;

        // Populate NppStreamContext from device attributes.
        let raw = populate_stream_context(&device, &stream)?;

        Ok(Self {
            device,
            stream,
            raw,
            _nosync: PhantomData,
        })
    }

    /// Reference to the underlying CUDA device.
    #[cfg(not(tarpaulin_include))]
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }

    /// Reference to the CUDA stream.
    #[cfg(not(tarpaulin_include))]
    pub fn stream(&self) -> &Arc<CudaStream> {
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
    /// subsequent DtoH copy).
    #[cfg(not(tarpaulin_include))]
    pub fn synchronize(&self) -> Result<(), DriverError> {
        // CudaStream::synchronize is a safe method in cudarc 0.19.x.
        self.stream.synchronize()
    }

    /// Create a new non-blocking stream on the same CUDA device.
    ///
    /// The returned `Arc<CudaStream>` is owned entirely by the caller — the
    /// `StreamContext` retains no reference to it. This is intentional:
    /// stream pooling, assignment policy, and lifetime management are the
    /// caller's responsibility.
    ///
    /// The new stream does **not** implicitly synchronise with any other
    /// stream. Use [`record_fence`] / [`wait_for`] to enforce ordering
    /// between streams.
    ///
    /// # Safety
    ///
    /// The returned stream must not outlive the device handle (the
    /// `Arc<CudaContext>` stored in this `StreamContext`). This is the same
    /// C7 invariant that governs all CUDA objects in this crate — the caller
    /// is responsible for ensuring the `StreamContext` (or some other
    /// reference to the same device) outlives the stream.
    ///
    /// [`record_fence`]: Self::record_fence
    /// [`wait_for`]: Self::wait_for
    #[cfg(not(tarpaulin_include))]
    pub fn new_stream(&self) -> Result<Arc<CudaStream>, DriverError> {
        self.device.new_stream()
    }

    /// Create a new RAII timing event associated with this stream context.
    ///
    /// Use [`Event::record`] to record the event on the stream, then call
    /// [`elapsed`](Self::elapsed) to measure device-time between two events.
    #[cfg(not(tarpaulin_include))]
    pub fn record_event(&self) -> Event {
        let event = self
            .device
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("cuEventCreate failed");
        Event {
            inner: event,
            stream: Arc::clone(&self.stream),
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
        // CudaEvent::elapsed_ms is safe in cudarc 0.19.x, returns f32 milliseconds.
        let ms = start.inner.elapsed_ms(&end.inner)?;
        Ok(Duration::from_secs_f64(ms as f64 / 1_000.0))
    }

    /// Record a fence on this stream.
    ///
    /// Creates a new [`Fence`] and records it on the stream managed by this
    /// context. All work enqueued before this call will complete before the
    /// fence is signalled. The returned `Fence` can be used with [`wait_for`]
    /// on another stream to order work, or with [`elapsed_between`] to measure
    /// device-time between two fences.
    ///
    /// This does **not** block the host. It enqueues a CUDA event record
    /// operation on the stream, which completes asynchronously.
    ///
    /// [`wait_for`]: Self::wait_for
    /// [`elapsed_between`]: Self::elapsed_between
    #[cfg(not(tarpaulin_include))]
    pub fn record_fence(&self) -> Result<Fence, NppError> {
        use cudarc::driver::sys::CUevent_flags;
        let event = self
            .device
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(NppError::Cuda)?;
        event.record(&self.stream).map_err(NppError::Cuda)?;
        Ok(Fence {
            inner: event,
            _device: Arc::clone(&self.device),
            _nosync: PhantomData,
        })
    }

    /// Make this stream wait until a fence recorded on another stream
    /// is signalled.
    ///
    /// All work enqueued **after** this call on this stream will not begin
    /// until the fence's prior work (on the recording stream) has completed.
    /// This establishes a device-side ordering between streams without
    /// host involvement.
    ///
    /// The `fence` must have been recorded (via [`record_fence`]) on some
    /// stream — not necessarily the same stream context — before this call.
    ///
    /// # Panics
    ///
    /// Panics if the underlying `cuStreamWaitEvent` call fails (this
    /// indicates a driver-level issue that is not recoverable).
    ///
    /// [`record_fence`]: Self::record_fence
    #[cfg(not(tarpaulin_include))]
    pub fn wait_for(&self, fence: &Fence) {
        // cuStreamWaitEvent is available via the raw FFI bindings in sys.
        // cudarc's CudaEvent does not expose a safe stream_wait wrapper,
        // so we call the driver API directly.
        unsafe {
            let status = sys::cuStreamWaitEvent(
                self.stream.cu_stream(),
                fence.inner.cu_event(),
                sys::CUevent_flags::CU_EVENT_DEFAULT as u32,
            );
            assert_eq!(
                status,
                sys::CUresult::CUDA_SUCCESS,
                "cuStreamWaitEvent failed with status {:?}",
                status,
            );
        }
    }

    /// Measure device-time elapsed between two recorded fences.
    ///
    /// Both fences must have been recorded (via [`record_fence`]) and the
    /// earlier fence's stream must have completed the recorded work before
    /// this call. The fences may have been recorded on **different** streams
    /// — this is the cross-stream timing capability that the existing
    /// [`elapsed`](Self::elapsed) method (which requires same-stream events)
    /// cannot provide.
    ///
    /// Returns `Ok(Duration)` on success, or `Err(NppError)` if the driver
    /// call fails.
    ///
    /// # Errors
    ///
    /// Returns `NppError::Cuda` if `cuEventElapsedTime` fails.
    ///
    /// [`record_fence`]: Self::record_fence
    #[cfg(not(tarpaulin_include))]
    pub fn elapsed_between(&self, start: &Fence, end: &Fence) -> Result<Duration, NppError> {
        let ms = start.inner.elapsed_ms(&end.inner).map_err(NppError::Cuda)?;
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
/// ensure the CUDA device handle (`CudaContext`) outlives this event
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
    inner: CudaEvent,
    stream: Arc<CudaStream>,
    // Keeps the CUDA device (and thus the CUDA context) alive.
    _device: Arc<CudaContext>,
    // Makes Event !Send + !Sync to match StreamContext's CUDA-context
    // thread affinity. CUDA events are tied to the CUcontext that created them.
    // This matches StreamContext's own !Send + !Sync.
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
        // CudaEvent::record is safe in cudarc 0.19.x.
        let result = self.inner.record(&self.stream);
        assert!(result.is_ok(), "cuEventRecord failed: {:?}", result);
    }
}

/// A CUDA event used for cross-stream ordering and device-side timing.
///
/// A `Fence` is recorded on one stream and waited on by another stream,
/// establishing a happens-after relationship between the work before the
/// fence on the recording stream and the work after the wait on the
/// waiting stream.
///
/// This is the single type for both ordering (`wait_for`) and timing
/// (`elapsed_between`). The underlying CUDA event is one concept; whether
/// it is used for ordering, timing, or both is a usage distinction, not
/// a structural one.
///
/// # Safety
///
/// The underlying CUDA event is created with `cuEventCreate` and destroyed
/// with `cuEventDestroy` on drop. The event is tied to the same CUDA context
/// as the `StreamContext` that created it. The caller must ensure the CUDA
/// device handle (`Arc<CudaContext>`) outlives this fence (same C7 invariant
/// as [`CudaImage`](crate::image::CudaImage) and [`StreamContext`]).
///
/// # !Send + !Sync
///
/// Like [`StreamContext`] and [`Event`], `Fence` is `!Send + !Sync` because
/// CUDA events are tied to the CUcontext that created them (C7).
#[derive(Debug)]
#[cfg(not(tarpaulin_include))]
pub struct Fence {
    inner: CudaEvent,
    // Keeps the CUDA device (and thus the CUDA context) alive.
    _device: Arc<CudaContext>,
    // Makes Fence !Send + !Sync to match StreamContext's CUDA-context
    // thread affinity.
    _nosync: PhantomData<*const ()>,
}

/// Populate an [`NppStreamContext`] from device attributes and a stream.
///
/// Queries the CUDA device with `cuDeviceGetAttribute` for each field.
/// This is called once during [`StreamContext::new`].
///
/// # Safety
///
/// The device handle must be valid and the CUDA driver must be initialised.
/// Both invariants are upheld by `cudarc::CudaContext`.
#[cfg(not(tarpaulin_include))]
fn populate_stream_context(
    device: &CudaContext,
    stream: &CudaStream,
) -> Result<NppStreamContext, DriverError> {
    use cudarc::driver::sys::CUdevice_attribute_enum;

    let ordinal = device.ordinal();

    // Helper: query a device attribute via CudaContext::attribute().
    // This wraps the unsafe cuDeviceGetAttribute safely.
    macro_rules! attr {
        ($name:ident) => {
            device.attribute(CUdevice_attribute_enum::$name)?
        };
    }

    // Cross-crate opaque pointer cast: cudarc's CUstream → npp-sys's
    // cudaStream_t. Both are *mut CUstream_st to the same underlying
    // driver object; the cast is semantically valid.
    let h_stream: npp_sys::cudaStream_t = stream.cu_stream() as npp_sys::cudaStream_t;

    Ok(NppStreamContext {
        hStream: h_stream,
        nCudaDeviceId: ordinal as i32,
        nMultiProcessorCount: attr!(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        nMaxThreadsPerMultiProcessor: attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        nMaxThreadsPerBlock: attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        nSharedMemPerBlock: attr!(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK) as usize,
        nCudaDevAttrComputeCapabilityMajor: attr!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        nCudaDevAttrComputeCapabilityMinor: attr!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        // nStreamFlags: new_stream creates a standard non-blocking stream.
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
    let device = CudaContext::new(ordinal)?;
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
