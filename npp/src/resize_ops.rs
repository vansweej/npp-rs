use crate::image::CudaImage;
use crate::imageops::*;
use npp_sys::{
    nppiResize_8u_C3R, NppiInterpolationMode_NPPI_INTER_CUBIC,
    NppiInterpolationMode_NPPI_INTER_LANCZOS, NppiInterpolationMode_NPPI_INTER_LINEAR,
    NppiInterpolationMode_NPPI_INTER_NN, NppiInterpolationMode_NPPI_INTER_SUPER, NppiRect,
    NppiSize,
};
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

#[inline(always)]
pub fn interpolation_mode(inter: ResizeInterpolation) -> i32 {
    match inter {
        ResizeInterpolation::NearestNeighbor => NppiInterpolationMode_NPPI_INTER_NN as i32,
        ResizeInterpolation::Linear => NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
        ResizeInterpolation::Cubic => NppiInterpolationMode_NPPI_INTER_CUBIC as i32,
        ResizeInterpolation::Super => NppiInterpolationMode_NPPI_INTER_SUPER as i32,
        ResizeInterpolation::Lanczos => NppiInterpolationMode_NPPI_INTER_LANCZOS as i32,
    }
}

impl CudaImage<u8> {
    pub fn resize(
        src: &CudaImage<u8>,
        dst: &mut CudaImage<u8>,
        inter: ResizeInterpolation,
    ) -> Result<(), CudaError> {
        debug_assert!(src.layout.channel_stride == 1 && dst.layout.channel_stride == 1);
        debug_assert!(src.layout.width_stride == 3 && dst.layout.width_stride == 3);

        let src_size: NppiSize = NppiSize {
            width: src.width() as i32,
            height: src.height() as i32,
        };
        let dst_size: NppiSize = NppiSize {
            width: dst.width() as i32,
            height: dst.height() as i32,
        };
        let src_rect: NppiRect = NppiRect {
            x: 0,
            y: 0,
            width: src.width() as i32,
            height: src.height() as i32,
        };
        let dst_rect: NppiRect = NppiRect {
            x: 0,
            y: 0,
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
            nppiResize_8u_C3R(
                src_ptr,
                src.layout.height_stride as i32,
                src_size,
                src_rect,
                dst_ptr,
                dst.layout.height_stride as i32,
                dst_size,
                dst_rect,
                interpolation_mode(inter),
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
    fn test_resize1() {
        let _ctx = initialize_cuda_device();
        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();
        let img_layout_src = img_src.as_rgb8().unwrap().sample_layout();

        let mut cuda_dst = match img_layout_src.channels {
            3 => CudaImage::new(640, 480, ColorType::Rgb8),
            _ => Err(CudaError::UnknownError),
        }
        .unwrap();

        let cuda_src = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

        CudaImage::resize(&cuda_src, &mut cuda_dst, ResizeInterpolation::Linear).unwrap();

        cuda_dst.save("resize1").unwrap();

        let img_dst = RgbImage::try_from(&cuda_dst).unwrap();

        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, 640);
        assert_eq!(img_layout_dst.height, 480);
    }

    #[test]
    fn test_resize2() {
        let _ctx = initialize_cuda_device();
        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();
        let img_layout_src = img_src.as_rgb8().unwrap().sample_layout();

        let mut cuda_dst = match img_layout_src.channels {
            3 => CudaImage::new(640, 480, ColorType::Rgb8),
            _ => Err(CudaError::UnknownError),
        }
        .unwrap();

        let cuda_src = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();
        let sub_cuda_src = cuda_src.sub_image(1722, 954, 510, 555).unwrap();

        CudaImage::resize(&sub_cuda_src, &mut cuda_dst, ResizeInterpolation::Linear).unwrap();

        sub_cuda_src.save("resize2").unwrap();

        let img_dst = RgbImage::try_from(&cuda_dst).unwrap();
        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, 640);
        assert_eq!(img_layout_dst.height, 480);
    }

    #[test]
    fn test_resize3() {
        let _ctx = initialize_cuda_device();
        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();
        let img_layout_src = img_src.as_rgb8().unwrap().sample_layout();

        let cuda_src = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();
        let sub_cuda_src1 = cuda_src.sub_image(1722, 954, 510, 555).unwrap();
        let mut sub_cuda_src2 = cuda_src.sub_image(10, 10, 510, 555).unwrap();

        CudaImage::resize(
            &sub_cuda_src1,
            &mut sub_cuda_src2,
            ResizeInterpolation::Linear,
        )
        .unwrap();

        cuda_src.save("resize3").unwrap();

        let img_dst = RgbImage::try_from(&sub_cuda_src2).unwrap();
        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, 510);
        assert_eq!(img_layout_dst.height, 555);
    }

    #[test]
    fn test_resize4() {
        let _ctx = initialize_cuda_device();
        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();
        let img_layout_src = img_src.as_rgb8().unwrap().sample_layout();

        let mut cuda_dst = match img_layout_src.channels {
            3 => CudaImage::new(640, 480, ColorType::Rgb8),
            _ => Err(CudaError::UnknownError),
        }
        .unwrap();

        let cuda_src = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

        CudaImage::resize(&cuda_src, &mut cuda_dst, ResizeInterpolation::Lanczos).unwrap();

        cuda_dst.save("lanczos").unwrap();

        let img_dst = RgbImage::try_from(&cuda_dst).unwrap();

        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, 640);
        assert_eq!(img_layout_dst.height, 480);
    }
}
