use crate::error::NppError;
use crate::image::{CudaImage, NppPixelType};

/// Interpolation methods supported by NPP resize operations.
///
/// Note: `Lanczos` is not supported for 16f channel types (NPP restriction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// `src` and `dst` must not overlap in memory. This applies to
/// **neighbourhood-gather** operations (e.g. resize samples a pixel window);
/// aliasing produces undefined results. Purely **elementwise** operations may
/// safely alias (see `Normalize`).
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
/// `src` and `dst` must not overlap in memory. This applies to
/// **neighbourhood-gather** operations; aliasing produces undefined results.
/// Purely **elementwise** operations may safely alias (see `Normalize`).
pub trait SwapChannels: Sized {
    /// Reorder channels from BGRA (4-channel) to RGB (3-channel).
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if src and dst dimensions disagree.
    /// Returns `NppError::Npp` if the underlying NPP call fails.
    fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError>;
}

/// Capability trait for NPP Mean (pixel-value average).
///
/// Returns one `f64` per channel representing the mean of all pixels in the
/// image. Internally uses the NPP two-call dance: `nppiMeanGetBufferHostSize_*`
/// to query scratch buffer size, then `nppiMean_*` to compute.
///
/// # Errors
///
/// Returns `NppError` if the underlying NPP call fails.
pub trait Mean: Sized {
    /// Compute per-channel mean pixel values.
    ///
    /// Returns a `Vec<f64>` with one entry per channel (e.g. 3 values for a
    /// 3-channel image).
    fn mean(&self) -> Result<Vec<f64>, NppError>;
}

/// Capability trait for cross-type pixel format conversion.
///
/// Converts `self` (source image) into a destination image of a different pixel type.
/// This is the crate's first cross-type operation family. Implemented only for
/// pixel type pairs that NPP supports for conversion (e.g. `u8 → f32`).
/// Unsupported `(src_type, dst_type)` pairs simply have no impl — calling them is
/// a compile-time error.
///
/// # Precondition
///
/// `src` and `dst` must not overlap in memory. This applies to
/// **neighbourhood-gather** operations; aliasing produces undefined results.
/// Purely **elementwise** operations may safely alias (see `Normalize`).
///
/// The destination image's `width`, `height`, and `channels` must match the source.
/// Mismatched dimensions result in `NppError::InvalidArgument`.
pub trait ConvertTo<Dst: NppPixelType> {
    /// Convert `self` into `dst`, performing type conversion and scaling as needed.
    ///
    /// The destination buffer is overwritten with the converted pixel values.
    /// The byte-step of the destination is computed from its element type
    /// (`size_of::<Dst>()`), not the source type.
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if `self` and `dst` dimensions or
    /// channel counts disagree, or if the conversion is not supported for the
    /// given channel count.
    /// Returns `NppError::Npp` if the underlying NPP call fails.
    fn convert(&self, dst: &mut CudaImage<Dst>) -> Result<(), NppError>;
}

/// Rounding modes for narrowing pixel conversions (`ConvertRounded`).
///
/// Controls how fractional source values are converted to integer
/// destination values. See NPP `NppRoundMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    /// Round to nearest; ties to **even** (`NPP_RND_NEAR`). E.g. 0.5→0, 1.5→2.
    Nearest,
    /// Round to nearest; ties **away from zero** (`NPP_RND_FINANCIAL`). E.g. 0.5→1, -1.5→-2.
    Financial,
    /// Truncate **toward zero** (`NPP_RND_ZERO`). E.g. 1.9→1, -2.5→-2.
    Zero,
}

/// Capability trait for narrowing cross-type conversion with explicit rounding.
///
/// Implemented only for `(src, dst)` pairs that NPP provides a rounding-mode
/// `nppiConvert_*` symbol for (narrowing conversions, e.g. `f32 → u8`).
/// Unsupported pairs simply have no impl — a compile-time error.
///
/// # Precondition
/// `self` and `dst` must not overlap; dimensions/channels must match (else
/// `NppError::InvalidArgument`). The CUDA device handle must outlive all
/// buffers (C7). This is an **owned-buffer** operation — no ROI sub-image
/// support (matching Convert/Normalize/Mean).
pub trait ConvertRounded<Dst: NppPixelType> {
    /// Convert `self` into `dst`, rounding fractional values per `mode`.
    ///
    /// # Errors
    /// `NppError::InvalidArgument` on dimension/channel/channel-count mismatch;
    /// `NppError::Npp` on NPP failure (including
    /// `NPP_ROUND_MODE_NOT_SUPPORTED_ERROR` if the pair rejects `mode`).
    fn convert_rounded(&self, dst: &mut CudaImage<Dst>, mode: RoundMode) -> Result<(), NppError>;
}

/// Capability trait for cross-type pixel normalization.
///
/// Normalizes `self` (source image) into a destination image of a different pixel type,
/// typically scaling to a standard range (e.g. `[0, 255] → [0.0, 1.0]` for neural network
/// preprocessing). Implemented only for pixel type pairs that NPP supports for
/// normalization (e.g. `u8 → f32`). Unsupported `(src_type, dst_type)` pairs simply
/// have no impl — calling them is a compile-time error.
///
/// # Precondition
///
/// `src` and `dst` must not overlap in memory. The convert step obeys the non-overlap
/// requirement for **neighbourhood-gather** operations; the scale step is purely
/// **elementwise** and may safely alias (see Phase 0 C4 wording).
///
/// The destination image's `width`, `height`, and `channels` must match the source.
/// Mismatched dimensions result in `NppError::InvalidArgument`.
pub trait Normalize<Dst: NppPixelType> {
    /// Normalize `self` into `dst`, performing type conversion and range scaling.
    ///
    /// For `u8 → f32`, this converts `[0, 255]` to `[0.0, 1.0]` by first calling
    /// `convert` (producing `[0.0, 255.0]`), then scaling by `1/255` in-place.
    ///
    /// The destination buffer is overwritten with the normalized pixel values.
    /// The byte-step of the destination is computed from its element type
    /// (`size_of::<Dst>()`), not the source type.
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if `self` and `dst` dimensions or
    /// channel counts disagree, or if the normalization is not supported for the
    /// given channel count.
    /// Returns `NppError::Npp` if the underlying NPP call fails.
    fn normalize(&self, dst: &mut CudaImage<Dst>) -> Result<(), NppError>;
}
