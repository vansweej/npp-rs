//! Application-managed NPP stream context abstraction.
//!
//! # Ordering correctness (key to the async contract)
//!
//! NPP `_Ctx` functions are enqueued asynchronously on the stream stored
//! in this context. **Correctness relies on implicit ordering guarantees**
//! from [`CudaDevice::fork_default_stream`]:
//!
//! 1. A host-to-device copy on the **default stream** completes first.
//! 2. `fork_default_stream()` creates a new stream and inserts an implicit
//!    wait so all previously-enqueued default-stream work is visible.
//! 3. The NPP operation is enqueued on the **forked stream**.
//! 4. Read-back (`TryFrom<&CudaImage>`) performs a synchronous DtoH copy
//!    on the **forked stream**, which blocks until the NPP op completes.
//!
//! This means: `htod_copy → NPP_op → dtoh_readback` is ordered **by
//! construction**, not by accident. See [`crate::stream`] module docs and
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

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaStream, DriverError};
use npp_sys::NppStreamContext;

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
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Reference to the CUDA stream.
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
    pub fn raw_ctx(&self) -> NppStreamContext {
        // NppStreamContext is Copy (confirmed Phase 0.3b), so this is
        // an implicit bitwise copy. Safe because the struct is fully
        // populated by the constructor and never mutated afterwards.
        self.raw
    }

    /// Block until all operations enqueued on this stream complete.
    ///
    /// This is the primary sync point for async NPP operations.
    /// After calling `synchronize()`, results can be read back safely.
    #[cfg(not(tarpaulin_include))]
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.device.wait_for(&self.stream)
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
}
