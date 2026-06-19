//! CUDA device initialization using `cudarc`.

use cudarc::driver::CudaDevice;
use std::sync::Arc;

use crate::error::NppError;

/// Initialize a CUDA device at the given ordinal and return a shared handle.
///
/// # Context-lifetime invariant (C7)
///
/// The returned `Arc<CudaDevice>` must be kept alive for the duration of any
/// `CudaImage` created from it. Dropping the device while buffers are live
/// results in `cuMemFree` against a destroyed context. cudarc's internal
/// `Arc<CudaDevice>` reference on every `CudaSlice` prevents this for the
/// common case.
pub fn initialize_cuda_device(ordinal: usize) -> Result<Arc<CudaDevice>, NppError> {
    let dev = CudaDevice::new(ordinal)?;
    Ok(dev)
}

/// Convenience wrapper: initialize device 0 (the default GPU).
pub fn default_cuda_device() -> Result<Arc<CudaDevice>, NppError> {
    initialize_cuda_device(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(feature = "gpu"), ignore)]
    #[test]
    fn test_cuda_initialize() {
        let result = default_cuda_device();
        assert!(result.is_ok());
    }
}
