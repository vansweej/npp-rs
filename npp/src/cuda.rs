//! CUDA device initialization using `cudarc`.

use cudarc::driver::CudaContext;
use std::sync::Arc;

use crate::error::NppError;

/// Initialize a CUDA device at the given ordinal and return a shared handle.
///
/// # Context-lifetime invariant (C7)
///
/// The returned `Arc<CudaContext>` must be kept alive for the duration of any
/// `CudaImage` created from it. Dropping the device while buffers are live
/// results in `cuMemFree` against a destroyed context. cudarc's internal
/// `Arc<CudaContext>` reference on every `CudaSlice` prevents this for the
/// common case.
///
/// NOTE: The cuda-12090 feature flag in Cargo.toml selects CUDA 12.9
/// specifically (Phase 0 verification). See docs/f11-plan.md.
#[cfg(not(tarpaulin_include))]
pub fn initialize_cuda_device(ordinal: usize) -> Result<Arc<CudaContext>, NppError> {
    let dev = CudaContext::new(ordinal)?;
    Ok(dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(feature = "gpu"), ignore)]
    #[test]
    fn test_cuda_initialize() {
        let result = initialize_cuda_device(0);
        assert!(result.is_ok());
    }
}
