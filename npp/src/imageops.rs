use crate::image::*;
use image::ColorType;
use npp_sys::{nppiResize_8u_C3R, NppiInterpolationMode_NPPI_INTER_LINEAR, NppiRect, NppiSize};
use rustacuda::error::*;

pub fn resize(src: &CudaImage<u8>, width: u32, height: u32) -> Result<CudaImage<u8>, CudaError> {
    let mut dst = match src.layout.channels {
        3 => CudaImage::<u8>::new(width, height, ColorType::Rgb8),
        _ => Err(CudaError::UnknownError),
    }?;

    let src_size: NppiSize = NppiSize {
        width: src.layout.width as i32,
        height: src.layout.height as i32,
    };
    let dst_size: NppiSize = NppiSize {
        width: dst.layout.width as i32,
        height: dst.layout.height as i32,
    };

    let src_rect: NppiRect = NppiRect {
        x: 0,
        y: 0,
        width: src.layout.width as i32,
        height: src.layout.height as i32,
    };
    let dst_rect: NppiRect = NppiRect {
        x: 0,
        y: 0,
        width: dst.layout.width as i32,
        height: dst.layout.height as i32,
    };

    let status = unsafe {
        nppiResize_8u_C3R(
            src.image_buf.as_ptr(),
            src.layout.height_stride as i32,
            src_size,
            src_rect,
            dst.image_buf.as_mut_ptr(),
            dst.layout.height_stride as i32,
            dst_size,
            dst_rect,
            NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
        )
    };
    if status == 0 {
        Ok(dst)
    } else {
        Err(CudaError::UnknownError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::io::Reader as ImageReader;
    use image::RgbImage;
    use rustacuda::prelude::*;
    use std::convert::TryFrom;
    #[test]
    fn test_resize() {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();
        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();
        let img_layout_src = img_src.as_rgb8().unwrap().sample_layout();

        let cuda_src = CudaImage::<u8>::try_from(img_src.as_rgb8().unwrap()).unwrap();
        let cuda_dst = resize(&cuda_src, 640, 480).unwrap();
        let img_dst = RgbImage::try_from(&cuda_dst).unwrap();

        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, 640);
        assert_eq!(img_layout_dst.height, 480);
    }
}
