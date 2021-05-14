use crate::layout::*;
use image::{ColorType, ImageBuffer, Rgb, RgbImage, RgbaImage};
use rustacuda::error::*;
use rustacuda::memory::*;
use std::cell::RefCell;
use std::convert::From;
use std::convert::TryFrom;
use std::mem::size_of;
use std::rc::Rc;

#[derive(Debug)]
pub struct CudaImage<T> {
    pub image_buf: Rc<RefCell<DeviceBuffer<T>>>,
    pub layout: CudaLayout,
}

impl<T> CudaImage<T> {
    pub fn new(width: u32, height: u32, ct: ColorType) -> Result<CudaImage<T>, CudaError> {
        let img_size_bytes =
            width as usize * height as usize * size_of::<T>() * ct.channel_count() as usize;
        let img_cuda_buffer = unsafe { DeviceBuffer::zeroed(img_size_bytes)? };
        let sl =
            CudaLayout::row_major_packed(size_of::<T>() as u8 * ct.channel_count(), width, height);
        Ok(CudaImage {
            image_buf: Rc::new(RefCell::new(img_cuda_buffer)),
            layout: sl,
        })
    }

    /// Get the index of the first point of the subimage
    fn get_index(&self, x: u32, y: u32) -> u32 {
        ((y - 1) * self.layout.height_stride as u32) + x
    }

    /// The bounding rectangle of this image.
    fn bounds(&self) -> (u32, u32, u32, u32) {
        (0, 0, self.layout.width, self.layout.height)
    }

    /// Returns true if this x, y coordinate is contained inside the sub image.
    /// Only width and height is used of the subimage
    fn in_sub_bounds(&self, x: u32, y: u32) -> bool {
        let (ix, iy, iw, ih) = self.bounds();
        x >= ix && x < ix + iw && y >= iy && y < iy + ih
    }

    /// Return true if this x, y coordinate is contained inside the sub image with
    /// regards to owning image.
    fn in_bounds(&self, x: u32, y: u32) -> bool {
        false
    }

    // pub fn sub_image(&self, x: u32, y: u32, width: u32, height: u32) -> Result(CudaImage<T>, CudaError>) {
    // // chech if x and y are inside img bounds
    // //if(x < 0 || x > self.layout,width)
    // // check if widht and heigh stay within image bounds
    //
    // // replace img_index and width and height of layout struct
    //
    // // return new sub_image
    // }
}

impl TryFrom<&RgbImage> for CudaImage<u8> {
    type Error = CudaError;

    fn try_from(img: &RgbImage) -> Result<Self, Self::Error> {
        let sl = img.sample_layout();
        let img_cuda_buffer = DeviceBuffer::from_slice(img.as_flat_samples().as_slice())?;
        Ok(CudaImage {
            image_buf: Rc::new(RefCell::new(img_cuda_buffer)),
            layout: CudaLayout::from(sl),
        })
    }
}

impl TryFrom<&RgbaImage> for CudaImage<u8> {
    type Error = CudaError;

    fn try_from(img: &RgbaImage) -> Result<Self, Self::Error> {
        let sl = img.sample_layout();
        let img_cuda_buffer = DeviceBuffer::from_slice(img.as_flat_samples().as_slice())?;
        Ok(CudaImage {
            image_buf: Rc::new(RefCell::new(img_cuda_buffer)),
            layout: CudaLayout::from(sl),
        })
    }
}

impl TryFrom<&CudaImage<u8>> for RgbImage {
    type Error = CudaError;

    fn try_from(di: &CudaImage<u8>) -> Result<Self, Self::Error> {
        let size = di.layout.width * di.layout.height * di.layout.channels as u32;
        let mut mem_host: Vec<u8> = Vec::with_capacity(size as usize);
        unsafe {
            mem_host.set_len(size as usize);
        }
        di.image_buf
            .borrow()
            .copy_to(&mut mem_host.as_mut_slice())?;
        let img_buf =
            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(di.layout.width, di.layout.height, mem_host);
        match img_buf {
            Some(x) => Ok(x),
            None => Err(CudaError::UnknownError), // TODO do we really want to map an image error to a CudaError
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::initialize_cuda_device;
    use image::io::Reader as ImageReader;
    use rustacuda::prelude::*;

    #[test]
    fn test_new() {
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

        let new_img =
            CudaImage::<u8>::new(img_layout.width, img_layout.height, ColorType::Rgb8).unwrap();

        assert_eq!(new_img.layout.channels, img_layout.channels);
        assert_eq!(new_img.layout.channel_stride, img_layout.channel_stride);
        assert_eq!(new_img.layout.width, img_layout.width);
        assert_eq!(new_img.layout.width_stride, img_layout.width_stride);
        assert_eq!(new_img.layout.height, img_layout.height);
        assert_eq!(new_img.layout.height_stride, img_layout.height_stride);
    }

    #[test]
    fn test_try_from_dynamic_image() {
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

        let cuda_buf = CudaImage::try_from(img.as_rgb8().unwrap()).unwrap();

        assert_eq!(cuda_buf.layout.channels, img_layout.channels);
        assert_eq!(cuda_buf.layout.channel_stride, img_layout.channel_stride);
        assert_eq!(cuda_buf.layout.width, img_layout.width);
        assert_eq!(cuda_buf.layout.width_stride, img_layout.width_stride);
        assert_eq!(cuda_buf.layout.height, img_layout.height);
        assert_eq!(cuda_buf.layout.height_stride, img_layout.height_stride);

        // might wanna add a Drop for the cuda_buf
    }

    #[test]
    fn test_try_from_cudaimage_image() {
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

        let cuda_buf = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

        let img_dst = RgbImage::try_from(&cuda_buf).unwrap();
        let img_layout_dst = img_dst.sample_layout();

        assert_eq!(img_layout_dst.channels, img_layout_src.channels);
        assert_eq!(img_layout_dst.channel_stride, img_layout_src.channel_stride);
        assert_eq!(img_layout_dst.width, img_layout_src.width);
        assert_eq!(img_layout_dst.width_stride, img_layout_src.width_stride);
        assert_eq!(img_layout_dst.height, img_layout_src.height);
        assert_eq!(img_layout_dst.height_stride, img_layout_src.height_stride);
    }

    #[test]
    fn test_get_index() {
        let _ctx = initialize_cuda_device();

        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();

        let cuda_buf = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();
        println!("{:?}", cuda_buf.layout);
        let sub_image_index = cuda_buf.get_index(10, 10);
        println!("{:?}", sub_image_index);
        assert_eq!(sub_image_index, 104554);
    }

    #[test]
    fn test_in_sub_bounds() {
        let _ctx = initialize_cuda_device();

        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();

        let cuda_buf = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

        assert_eq!(cuda_buf.in_sub_bounds(10, 10), true);
    }
}
