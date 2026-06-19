use crate::layout::CudaLayout;
use crate::error::NppError;
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, CudaViewMut};
use std::convert::TryFrom;
use std::sync::Arc;

/// Marker trait for NPP primitive pixel element types. Every concrete NPP
/// element type implements this trait. The trait is sealed — it cannot be
/// implemented outside this module.
///
/// Types implementing `NppPixelType` are:
///
/// | NPP name | Rust type | Bits |
/// |----------|-----------|------|
/// | `8u`     | `u8`      | 8    |
/// | `8s`     | `i8`      | 8    |
/// | `16u`    | `u16`     | 16   |
/// | `16s`    | `i16`     | 16   |
/// | `32u`    | `u32`     | 32   |
/// | `32s`    | `i32`     | 32   |
/// | `32f`    | `f32`     | 32   |
/// | `64f`    | `f64`     | 64   |
/// | `16f`    | `half::f16` | 16 |  (requires `half` crate)
///
/// All types are constructible (allocatable + zeroable) in M1. Operation
/// capability is expressed by separate traits (e.g. `Resize`, `SwapChannels`);
/// an unsupported `(type, op)` pair simply has no trait impl, making it a
/// compile-time error rather than a runtime `NotImplemented`.
pub trait NppPixelType: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy + private::Sealed {
    /// Bit-width of the element type (e.g. 8 for `u8`, 32 for `f32`).
    const BITS: u8;
}

mod private {
    pub trait Sealed {}
}

impl private::Sealed for u8 {}
impl NppPixelType for u8 {
    const BITS: u8 = 8;
}

impl private::Sealed for i8 {}
impl NppPixelType for i8 {
    const BITS: u8 = 8;
}

impl private::Sealed for u16 {}
impl NppPixelType for u16 {
    const BITS: u8 = 16;
}

impl private::Sealed for i16 {}
impl NppPixelType for i16 {
    const BITS: u8 = 16;
}

impl private::Sealed for u32 {}
impl NppPixelType for u32 {
    const BITS: u8 = 32;
}

impl private::Sealed for i32 {}
impl NppPixelType for i32 {
    const BITS: u8 = 32;
}

impl private::Sealed for f32 {}
impl NppPixelType for f32 {
    const BITS: u8 = 32;
}

impl private::Sealed for f64 {}
impl NppPixelType for f64 {
    const BITS: u8 = 64;
}

// TODO(M2): 16f via half crate

/// A GPU-resident N-dimensional image buffer backed by a contiguous NPP-compatible
/// device allocation. `T` is the element type (see `NppPixelType`).
///
/// # Device lifetime invariant (C7)
///
/// The `Arc<CudaDevice>` stored in every `CudaImage` must outlive all operations
/// on the image. Dropping the device while a `CudaImage` is live results in
/// `cuMemFree` against a destroyed context. This is enforced for free by cudarc's
/// internal `Arc<CudaDevice>` reference on every `CudaSlice`.
///
/// # Thread safety
///
/// `CudaImage<T>` is `Send + Sync` because `CudaSlice<T>` is `Send + Sync` in
/// cudarc 0.9. However, CUDA contexts are thread-bound; safe cross-thread usage
/// requires explicit context management (deferred to M2 — see F8).
#[derive(Debug)]
pub struct CudaImage<T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) buf: CudaSlice<T>,
    pub(crate) layout: CudaLayout,
}

impl<T: NppPixelType> CudaImage<T> {
    /// Allocate a new GPU image with the given dimensions and channel count.
    ///
    /// # Arguments
    ///
    /// * `device` - The CUDA device to allocate on
    /// * `channels` - Number of channels per pixel
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Errors
    ///
    /// Returns `NppError::Cuda` if device allocation fails.
    pub fn new(
        device: Arc<CudaDevice>,
        channels: u8,
        width: u32,
        height: u32,
    ) -> Result<Self, NppError> {
        let num_elements = (width as usize) * (height as usize) * (channels as usize);
        let buf = device.alloc_zeros::<T>(num_elements)?;
        let layout = CudaLayout::row_major_packed(channels, width, height);
        Ok(CudaImage {
            device,
            buf,
            layout,
        })
    }

    /// Create a GPU image from host data.
    ///
    /// # Arguments
    ///
    /// * `device` - The CUDA device to allocate on
    /// * `channels` - Number of channels per pixel
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - Host data buffer (must have length == width * height * channels)
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if data length does not match dimensions.
    /// Returns `NppError::Cuda` if device allocation or copy fails.
    pub fn from_host(
        device: Arc<CudaDevice>,
        channels: u8,
        width: u32,
        height: u32,
        data: &[T],
    ) -> Result<Self, NppError> {
        let expected_len = (width as usize) * (height as usize) * (channels as usize);
        if data.len() != expected_len {
            return Err(NppError::InvalidArgument(format!(
                "data length {} does not match dimensions {}x{}x{}={}",
                data.len(),
                width,
                height,
                channels,
                expected_len
            )));
        }
        let buf = device.htod_sync_copy(data)?;
        let layout = CudaLayout::row_major_packed(channels, width, height);
        Ok(CudaImage {
            device,
            buf,
            layout,
        })
    }

    /// Get the image dimensions as (channels, width, height).
    pub fn dimensions(&self) -> (u8, u32, u32) {
        self.layout.dimensions()
    }

    /// Get the image width in pixels.
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the image height in pixels.
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// Get the number of channels per pixel.
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Compute the linear index for a pixel at (x, y).
    pub fn get_index(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.layout.height_stride as usize)
            + (x as usize) * (self.layout.width_stride as usize)
    }

    /// Get the (x, y) coordinates of the first pixel in the image.
    pub fn get_start_point(&self) -> (u32, u32) {
        let img_index = self.layout.img_index as usize;
        let x = (img_index / (self.layout.channels as usize)) as u32;
        let y = 0;
        (x, y)
    }

    /// Get the bounding box of the image as (x_min, y_min, x_max, y_max).
    pub fn bounds(&self) -> (u32, u32, u32, u32) {
        (0, 0, self.layout.width, self.layout.height)
    }

    /// Check if a pixel coordinate (x, y) is within the image bounds.
    pub fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.layout.width && y < self.layout.height
    }

    /// Create a borrowed sub-image view.
    ///
    /// # Arguments
    ///
    /// * `x` - Left edge of the sub-image
    /// * `y` - Top edge of the sub-image
    /// * `w` - Width of the sub-image
    /// * `h` - Height of the sub-image
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if the sub-image bounds exceed the parent image.
    pub fn sub_image(
        &self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<CudaImageView<'_, T>, NppError> {
        if x + w > self.layout.width || y + h > self.layout.height {
            return Err(NppError::InvalidArgument(format!(
                "sub-image bounds ({}, {}, {}, {}) exceed parent bounds ({}, {})",
                x, y, w, h, self.layout.width, self.layout.height
            )));
        }

        let img_index = self.get_index(x, y);
        let start = self.layout.img_index as usize + img_index;
        let len = (h as usize) * (self.layout.height_stride as usize);

        let view = self.buf.slice(start..start + len);
        let layout = CudaLayout {
            channels: self.layout.channels,
            channel_stride: 1,
            width: w,
            width_stride: self.layout.width_stride,
            height: h,
            height_stride: self.layout.height_stride,
            img_index: 0,
        };

        Ok(CudaImageView {
            device: self.device.clone(),
            view,
            layout,
        })
    }

    /// Create a mutable borrowed sub-image view.
    ///
    /// # Arguments
    ///
    /// * `x` - Left edge of the sub-image
    /// * `y` - Top edge of the sub-image
    /// * `w` - Width of the sub-image
    /// * `h` - Height of the sub-image
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if the sub-image bounds exceed the parent image.
    pub fn sub_image_mut(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<CudaImageViewMut<'_, T>, NppError> {
        if x + w > self.layout.width || y + h > self.layout.height {
            return Err(NppError::InvalidArgument(format!(
                "sub-image bounds ({}, {}, {}, {}) exceed parent bounds ({}, {})",
                x, y, w, h, self.layout.width, self.layout.height
            )));
        }

        let img_index = self.get_index(x, y);
        let start = self.layout.img_index as usize + img_index;
        let len = (h as usize) * (self.layout.height_stride as usize);

        let view = self.buf.slice_mut(start..start + len);
        let layout = CudaLayout {
            channels: self.layout.channels,
            channel_stride: 1,
            width: w,
            width_stride: self.layout.width_stride,
            height: h,
            height_stride: self.layout.height_stride,
            img_index: 0,
        };

        Ok(CudaImageViewMut {
            device: self.device.clone(),
            view,
            layout,
        })
    }
}

/// A borrowed read-only view of a GPU image.
#[derive(Debug)]
pub struct CudaImageView<'a, T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) view: CudaView<'a, T>,
    pub(crate) layout: CudaLayout,
}

impl<'a, T: NppPixelType> CudaImageView<'a, T> {
    /// Get the image width in pixels.
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the image height in pixels.
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// Get the number of channels per pixel.
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Get the image dimensions as (channels, width, height).
    pub fn dimensions(&self) -> (u8, u32, u32) {
        self.layout.dimensions()
    }

    /// Get the (x, y) coordinates of the first pixel in the view.
    pub fn get_start_point(&self) -> (u32, u32) {
        let img_index = self.layout.img_index as usize;
        let x = (img_index / (self.layout.channels as usize)) as u32;
        let y = 0;
        (x, y)
    }

    /// Extract the raw device pointer for NPP calls.
    ///
    /// See `docs/spike-cudarc-ptr-bridge.md` for the authoritative pattern.
    pub(crate) fn device_ptr(&self) -> *const T {
        let cu_ptr = cudarc::driver::DevicePtr::device_ptr(&self.view);
        *cu_ptr as *const T
    }
}

/// A borrowed mutable view of a GPU image.
#[derive(Debug)]
pub struct CudaImageViewMut<'a, T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) view: CudaViewMut<'a, T>,
    pub(crate) layout: CudaLayout,
}

impl<'a, T: NppPixelType> CudaImageViewMut<'a, T> {
    /// Get the image width in pixels.
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the image height in pixels.
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// Get the number of channels per pixel.
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Get the image dimensions as (channels, width, height).
    pub fn dimensions(&self) -> (u8, u32, u32) {
        self.layout.dimensions()
    }

    /// Get the (x, y) coordinates of the first pixel in the view.
    pub fn get_start_point(&self) -> (u32, u32) {
        let img_index = self.layout.img_index as usize;
        let x = (img_index / (self.layout.channels as usize)) as u32;
        let y = 0;
        (x, y)
    }

    /// Extract the raw mutable device pointer for NPP calls.
    ///
    /// See `docs/spike-cudarc-ptr-bridge.md` for the authoritative pattern.
    pub(crate) fn device_ptr_mut(&mut self) -> *mut T {
        let cu_ptr = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut self.view);
        *cu_ptr as *mut T
    }
}

/// Copy a GPU image to host memory.
impl<T: NppPixelType> TryFrom<&CudaImage<T>> for Vec<T> {
    type Error = NppError;

    fn try_from(img: &CudaImage<T>) -> Result<Self, Self::Error> {
        let host: Vec<T> = img.device.dtoh_sync_copy(&img.buf)?;
        Ok(host)
    }
}
