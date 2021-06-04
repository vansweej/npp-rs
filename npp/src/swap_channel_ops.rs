use crate::image::CudaImage;
use crate::imageops::*;
use npp_sys::{nppiSwapChannels_8u_C4C3R, NppiSize};
use rustacuda::error::*;
use std::os::raw::c_int;

impl SwapChannels<u8> for CudaImage<u8> {
    fn bgra_to_rgb(src: &CudaImage<u8>, dst: &mut CudaImage<u8>) -> Result<(), CudaError> {
        debug_assert!(src.layout.channel_stride == 1 && dst.layout.channel_stride == 1);
        debug_assert!(src.layout.width_stride == 4 && dst.layout.width_stride == 3);
        debug_assert!(
            src.layout.width == dst.layout.width && src.layout.height == dst.layout.height
        );

        let nppi_size: NppiSize = NppiSize {
            width: dst.width() as i32,
            height: dst.height() as i32,
        };
        let src_ptr = unsafe {
            src.image_buf
                .borrow_mut()
                .as_device_ptr()
                .offset(src.layout.img_index as isize)
                .as_raw()
        };
        let dst_ptr = unsafe {
            dst.image_buf
                .borrow_mut()
                .as_device_ptr()
                .offset(dst.layout.img_index as isize)
                .as_raw_mut()
        };
        let status = unsafe {
            let order: [c_int; 3] = [2, 1, 0];
            nppiSwapChannels_8u_C4C3R(
                src_ptr,
                src.layout.height_stride as i32,
                dst_ptr,
                dst.layout.height_stride as i32,
                nppi_size,
                &order[0],
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(CudaError::UnknownError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::initialize_cuda_device;
    use crate::image::Persistable;
    use image::io::Reader as ImageReader;
    use image::{ColorType, RgbImage};
    use std::convert::TryFrom;

    #[test]
    fn test_bgra_to_rgb() {
        let _ctx = initialize_cuda_device();
        let img_src = ImageReader::open("test_resources/bgraimg.png")
            .unwrap()
            .decode()
            .unwrap();

        let img_layout_src = img_src.as_rgba8().unwrap().sample_layout();

        let mut cuda_dst =
            CudaImage::<u8>::new(img_layout_src.width, img_layout_src.height, ColorType::Rgb8)
                .unwrap();

        let cuda_src = CudaImage::try_from(&img_src.to_rgba8()).unwrap();

        CudaImage::bgra_to_rgb(&cuda_src, &mut cuda_dst).unwrap();

        cuda_dst.save("bgra2rgba").unwrap();

        let img_dst = RgbImage::try_from(&cuda_dst).unwrap();

        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, 3);
        assert_eq!(img_layout_dst.width, img_layout_src.width);
        assert_eq!(img_layout_dst.height, img_layout_src.height);
    }
}
