use crate::image::CudaImage;
use rustacuda::error::*;

// pub trait ResizeImage<T> {
//     fn resize(
//         src: &CudaImage<T>,
//         dst: &mut CudaImage<T>,
//         inter: ResizeInterpolation,
//     ) -> Result<(), CudaError>;
// }

pub trait SwapChannels<T> {
    fn bgra_to_rgb(src: &CudaImage<T>, dst: &mut CudaImage<T>) -> Result<(), CudaError>;
}
