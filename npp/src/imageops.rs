use crate::image::CudaImage;
use rustacuda::error::*;

/// Interpolation methods supported in resize calls
/// 16f channel types do not support Lanczos
pub enum ResizeInterpolation {
    NearestNeighbor,
    Linear,
    Cubic,
    Super,
    Lanczos,
}

pub trait ResizeImage<T> {
    fn resize(
        src: &CudaImage<T>,
        dst: &mut CudaImage<T>,
        inter: ResizeInterpolation,
    ) -> Result<(), CudaError>;
}
