use crate::error::NppError;

/// Interpolation methods supported by NPP resize operations.
///
/// Note: `Lanczos` is not supported for 16f channel types (NPP restriction).
#[derive(Debug, Clone, Copy)]
pub enum ResizeInterpolation {
    /// Nearest-neighbor interpolation (no anti-aliasing).
    NearestNeighbor,
    /// Linear interpolation.
    Linear,
    /// Cubic convolution interpolation.
    Cubic,
    /// Super-sampling interpolation.
    Super,
    /// Lanczos interpolation.
    Lanczos,
}

/// Capability trait for NPP resize operations.
///
/// Implemented only for pixel types that NPP supports for resize.
/// Unsupported `(type, op)` pairs simply have no impl — calling them is
/// a compile-time error.
///
/// # Precondition
///
/// `src` and `dst` must refer to **non-overlapping** device buffers.
/// Passing overlapping ROIs (e.g. two sub-views of the same parent image)
/// to resize is undefined behavior in NPP.
pub trait Resize: Sized {
    /// Resize `self` into `dst` using the given interpolation method.
    ///
    /// # Errors
    ///
    /// Returns `NppError` if the underlying NPP call fails.
    fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError>;
}

/// Capability trait for 4-channel BGRA → 3-channel RGB channel reordering.
///
/// Same impl/non-impl model as `Resize`. Current M1 scope: `CudaImage<u8>` only.
///
/// # Precondition
///
/// `src` and `dst` must refer to **non-overlapping** device buffers.
pub trait SwapChannels: Sized {
    /// Reorder channels from BGRA (4-channel) to RGB (3-channel).
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if src and dst dimensions disagree.
    /// Returns `NppError::Npp` if the underlying NPP call fails.
    fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError>;
}
