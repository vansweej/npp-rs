use crate::layout::*;
use image::{Bgra, ColorType, ImageBuffer, Rgb, RgbImage, RgbaImage};
use rustacuda::error::*;
use rustacuda::memory::*;
use std::cell::RefCell;
use std::convert::From;
use std::convert::TryFrom;
use std::env::temp_dir;
use std::mem::size_of;
use std::rc::Rc;

#[derive(Debug)]
pub struct CudaImage<T> {
    pub(in crate) image_buf: Rc<RefCell<DeviceBuffer<T>>>,
    pub(in crate) layout: CudaLayout,
}

pub trait Persistable {
    fn save(&self, filename: &str) -> Result<(), image::ImageError>;
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

    /// The width and height of this image.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.layout.width, self.layout.height)
    }

    /// The width of this image.
    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// The height of this image.
    #[inline(always)]
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// number of channels of this image
    #[inline(always)]
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Get the index of the first point of the subimage
    #[inline(always)]
    fn get_index(&self, x: u32, y: u32) -> usize {
        ((y * self.layout.height_stride as u32) + (x * self.layout.width_stride as u32)) as usize
    }

    /// get the x, y coordinates from the img_index
    #[inline(always)]
    pub fn get_start_point(&self) -> (u32, u32) {
        let y = self.layout.img_index / self.layout.height_stride;
        let x = (self.layout.img_index % self.layout.height_stride) / self.layout.channels as usize;
        (x as u32, y as u32)
    }

    /// The bounding rectangle of this image.
    #[inline(always)]
    fn bounds(&self) -> (u32, u32, u32, u32) {
        (0, 0, self.layout.width, self.layout.height)
    }

    /// Returns true if this x, y coordinate is contained inside the sub image.
    /// Only width and height is used of the subimage
    #[inline(always)]
    fn in_bounds(&self, x: u32, y: u32) -> bool {
        let (ix, iy, iw, ih) = self.bounds();
        x >= ix && x <= ix + iw && y >= iy && y <= iy + ih
    }

    /// returns a sub image from self. x and y are 0 based indexed. The cuda buffers
    /// reference count is incremented but not copied.
    pub fn sub_image(&self, x: u32, y: u32, w: u32, h: u32) -> Result<CudaImage<T>, CudaError> {
        if self.in_bounds(x, y) && self.in_bounds(x + w, y + h) {
            let lay = CudaLayout {
                channels: self.layout.channels,
                channel_stride: self.layout.channel_stride,
                width: w,
                width_stride: self.layout.width_stride,
                height: h,
                height_stride: self.layout.height_stride,
                img_index: self.get_index(x, y),
            };
            Ok(CudaImage {
                image_buf: Rc::clone(&self.image_buf),
                layout: lay,
            })
        } else {
            Err(CudaError::UnknownError)
        }
    }
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

impl TryFrom<&ImageBuffer<Bgra<u8>, Vec<u8>>> for CudaImage<u8> {
    type Error = CudaError;

    fn try_from(img: &ImageBuffer<Bgra<u8>, Vec<u8>>) -> Result<Self, Self::Error> {
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

        let mut mem_host_iter =
            mem_host.chunks_mut(di.layout.width as usize * di.layout.width_stride);

        for row_index in 0..di.layout.height {
            let ptr = unsafe {
                di.image_buf.borrow_mut().as_device_ptr().offset(
                    (di.layout.img_index + (row_index as usize * di.layout.height_stride)) as isize,
                )
            };
            let slice = unsafe {
                DeviceSlice::from_raw_parts(ptr, di.layout.width as usize * di.layout.width_stride)
            };
            let mem_host_slice = mem_host_iter.next().unwrap();

            slice.copy_to(mem_host_slice)?;
        }

        let img_buf =
            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(di.layout.width, di.layout.height, mem_host);
        match img_buf {
            Some(x) => Ok(x),
            None => Err(CudaError::UnknownError), // TODO do we really want to map an image error to a CudaError
        }
    }
}

impl Persistable for CudaImage<u8> {
    fn save(&self, filename: &str) -> Result<(), image::ImageError> {
        let mut tmp_dir = temp_dir();
        tmp_dir.push(filename);
        tmp_dir.set_extension("png");
        let path = tmp_dir.to_str().unwrap();

        let sub_image_dst = RgbImage::try_from(self).unwrap();
        sub_image_dst.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::initialize_cuda_device;
    use image::io::Reader as ImageReader;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_new() {
        let _ctx = initialize_cuda_device();
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
        let _ctx = initialize_cuda_device();
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
        let _ctx = initialize_cuda_device();

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

        let image = CudaImage::<u8>::new(100, 100, ColorType::Rgb8).unwrap();
        let sub_image_index = image.get_index(0, 0);
        assert_eq!(sub_image_index, 0);
        let sub_image_index = image.get_index(10, 10);
        assert_eq!(sub_image_index, 3030);
    }

    #[test]
    fn test_in_bounds() {
        let _ctx = initialize_cuda_device();

        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();

        let cuda_buf = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

        assert_eq!(cuda_buf.in_bounds(10, 10), true);
        assert_eq!(cuda_buf.in_bounds(3972, 500), false);
    }

    #[test]
    fn test_sub_image() {
        let _ctx = initialize_cuda_device();

        let image1 = CudaImage::<u8>::new(100, 100, ColorType::Rgb8).unwrap();
        let sub_image1 = image1.sub_image(5, 5, 10, 10).unwrap();

        image1.save("image1").unwrap();
        sub_image1.save("sub_image1").unwrap();

        assert_eq!(sub_image1.width(), 10);
        assert_eq!(sub_image1.height(), 10);
    }

    #[test]
    fn test_sub_image2() {
        let _ctx = initialize_cuda_device();

        let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
            .unwrap()
            .decode()
            .unwrap();

        let cuda_buf = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

        let sub_image2 = cuda_buf.sub_image(1722, 954, 510, 555).unwrap();

        sub_image2.save("sub_image2").unwrap();

        assert_eq!(sub_image2.dimensions(), (510, 555));
    }

    #[test]
    fn test_get_start_point() {
        let _ctx = initialize_cuda_device();

        let image1 = CudaImage::<u8>::new(100, 100, ColorType::Rgb8).unwrap();
        let sub_image1 = image1.sub_image(5, 5, 10, 10).unwrap();

        assert_eq!(image1.get_start_point(), (0, 0));

        assert_eq!(sub_image1.layout.height_stride, 300);
        assert_eq!(sub_image1.layout.img_index, 1515);
        assert_eq!(sub_image1.get_start_point(), (5, 5));
    }
}
