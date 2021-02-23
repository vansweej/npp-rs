use image::flat::SampleLayout;
use image::*;
use rustacuda::memory::*;

pub struct CudaImage<T> {
    image_buf: DeviceBuffer<T>,
    layout: SampleLayout,
}

impl From<&image::DynamicImage> for CudaImage<u8> {
    fn from(img: &image::DynamicImage) -> Self {
        match img {
            DynamicImage::ImageRgb8(rgb_image) => {
                let sl = rgb_image.sample_layout();
                let img_size_bytes = sl.width * sl.height * sl.channels as u32;
                let mut img_cuda_buffer =
                    unsafe { DeviceBuffer::uninitialized(img_size_bytes as usize).unwrap() };
                img_cuda_buffer
                    .copy_from(rgb_image.as_flat_samples().as_slice())
                    .unwrap();
                CudaImage {
                    image_buf: img_cuda_buffer,
                    layout: sl,
                }
            }
            _ => CudaImage {
                image_buf: unsafe { DeviceBuffer::zeroed(0).unwrap() },
                layout: SampleLayout {
                    width: 0,
                    height: 0,
                    channel_stride: 0,
                    height_stride: 0,
                    width_stride: 0,
                    channels: 0,
                },
            },
        }
    }
}

impl From<&CudaImage<u8>> for image::DynamicImage {
    fn from(di: &CudaImage<u8>) -> Self {
        let size = di.layout.width * di.layout.height * di.layout.channels as u32;
        let mut mem_host: Vec<u8> = Vec::with_capacity(size as usize);
        unsafe {
            mem_host.set_len(size as usize);
        }
        di.image_buf.copy_to(&mut mem_host.as_mut_slice()).unwrap();
        let img_buf =
            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(di.layout.width, di.layout.height, mem_host)
                .unwrap();
        DynamicImage::ImageRgb8(img_buf)
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

        let cuda_buf = CudaImage::from(&img);

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
        let img = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();

        let cuda_buf = CudaImage::from(&img);

        let new_image = DynamicImage::from(&cuda_buf);

        new_image.save("/tmp/test123.jpg").unwrap();
    }
}
