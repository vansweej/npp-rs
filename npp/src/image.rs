use image::flat::SampleLayout;
use image::*;
use rustacuda::error::*;
use rustacuda::memory::*;
use std::convert::TryFrom;

pub struct CudaImage<T> {
    image_buf: DeviceBuffer<T>,
    layout: SampleLayout,
    color_type: ColorType,
}

impl TryFrom<&image::DynamicImage> for CudaImage<u8> {
    type Error = CudaError;
    fn try_from(img: &image::DynamicImage) -> Result<Self, Self::Error> {
        match img {
            DynamicImage::ImageRgb8(rgb_image) => {
                let sl = rgb_image.sample_layout();
                let img_size_bytes = sl.width * sl.height * sl.channels as u32;
                let mut img_cuda_buffer =
                    unsafe { DeviceBuffer::uninitialized(img_size_bytes as usize)? };
                img_cuda_buffer.copy_from(rgb_image.as_flat_samples().as_slice())?;
                Ok(CudaImage {
                    image_buf: img_cuda_buffer,
                    layout: sl,
                    color_type: ColorType::Rgb8,
                })
            }
            _ => Err(CudaError::InvalidImage),
        }
    }
}

impl TryFrom<&CudaImage<u8>> for image::DynamicImage {
    type Error = CudaError;
    fn try_from(di: &CudaImage<u8>) -> Result<Self, Self::Error> {
        let size = di.layout.width * di.layout.height * di.layout.channels as u32;
        let mut mem_host: Vec<u8> = Vec::with_capacity(size as usize);
        unsafe {
            mem_host.set_len(size as usize);
        }
        di.image_buf.copy_to(&mut mem_host.as_mut_slice())?;
        let img_buf = match di.color_type {
            ColorType::Rgb8 => ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
                di.layout.width,
                di.layout.height,
                mem_host,
            ),
            _ => None,
        };
        match img_buf {
            Some(x) => Ok(DynamicImage::ImageRgb8(x)),
            None => Err(CudaError::UnknownError), // TODO do we really want to map an image error to a CudaError
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::io::Reader as ImageReader;
    use rustacuda::prelude::*;
    #[test]
    fn test_from_dynamic_image() {
        rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();
        let img = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();
        let img_layout = img.as_rgb8().unwrap().sample_layout();

        let cuda_buf = CudaImage::try_from(&img).unwrap();

        assert_eq!(cuda_buf.layout.channels, img_layout.channels);
        assert_eq!(cuda_buf.layout.channel_stride, img_layout.channel_stride);
        assert_eq!(cuda_buf.layout.width, img_layout.width);
        assert_eq!(cuda_buf.layout.width_stride, img_layout.width_stride);
        assert_eq!(cuda_buf.layout.height, img_layout.height);
        assert_eq!(cuda_buf.layout.height_stride, img_layout.height_stride);

        // might wanna add a Drop for the cuda_buf
    }

    #[test]
    fn test_from_cudaimage_image() {
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

        let cuda_buf = CudaImage::try_from(&img_src).unwrap();

        let img_dst = DynamicImage::try_from(&cuda_buf).unwrap();
        let img_layout_dst = img_dst.as_rgb8().unwrap().sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, img_layout_src.width);
        assert_eq!(img_layout_dst.width_stride, img_layout_src.width_stride);
        assert_eq!(img_layout_dst.height, img_layout_src.height);
        assert_eq!(img_layout_dst.height_stride, img_layout_src.height_stride);
    }
}
