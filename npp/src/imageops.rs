use crate::image::CudaImage;
use rustacuda::error::*;

pub trait SwapChannels<T> {
    fn bgra_to_rgb(src: &CudaImage<T>, dst: &mut CudaImage<T>) -> Result<(), CudaError>;
}
