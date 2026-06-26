#![allow(clippy::arc_with_non_send_sync)]

use crate::error::NppError;
use crate::layout::CudaLayout;
use crate::stream::StreamContext;
use cudarc::driver::{CudaContext, CudaSlice, CudaView, CudaViewMut};
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
pub trait NppPixelType:
    cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy + private::Sealed
{
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
/// The [`StreamContext`] stored in every `CudaImage` must outlive all operations
/// on the image. The underlying [`CudaDevice`] handle is reference-counted via
/// [`Arc`], so it stays alive as long as any `CudaImage` or `StreamContext` holds
/// a reference.
///
/// # Thread safety
///
/// `CudaImage<T>` is `!Send + !Sync` because [`StreamContext`] contains a
/// `CudaStream` which is thread-bound (CUDA streams cannot safely be used
/// from multiple threads). Images created from the same `StreamContext` share
/// the same forked stream; cross-thread usage requires external synchronisation.
#[derive(Debug)]
pub struct CudaImage<T: NppPixelType> {
    pub(crate) ctx: Arc<StreamContext>,
    pub(crate) buf: CudaSlice<T>,
    pub(crate) layout: CudaLayout,
}

/// Validate image dimensions before allocation.
///
/// Returns `NppError::InvalidArgument` if any of `channels`, `width`, or
/// `height` is zero. A zero-dimension image allocates a degenerate buffer that
/// NPP later rejects with a cryptic `NPP_SIZE_ERROR`; validating here surfaces
/// the mistake eagerly at the construction site instead.
///
/// Pure-logic (no GPU), so it is unit-tested on the host and counts toward
/// coverage — unlike the `#[cfg(not(tarpaulin_include))]` constructors that
/// call it.
fn validate_dims(channels: u8, width: u32, height: u32) -> Result<(), NppError> {
    if width == 0 || height == 0 || channels == 0 {
        return Err(NppError::InvalidArgument(format!(
            "image dimensions must be non-zero, got {}x{}x{}",
            width, height, channels
        )));
    }
    Ok(())
}

impl<T: NppPixelType> CudaImage<T> {
    /// Allocate a new GPU image with the given dimensions and channel count.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The stream context providing the CUDA device for allocation
    /// * `channels` - Number of channels per pixel
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if any dimension (width, height, or
    /// channels) is zero.
    /// Returns `NppError::Cuda` if device allocation fails.
    #[cfg(not(tarpaulin_include))]
    pub fn new(
        ctx: Arc<StreamContext>,
        channels: u8,
        width: u32,
        height: u32,
    ) -> Result<Self, NppError> {
        validate_dims(channels, width, height)?;
        let num_elements = (width as usize) * (height as usize) * (channels as usize);
        let buf = ctx.stream().alloc_zeros::<T>(num_elements)?;
        let layout = CudaLayout::row_major_packed(channels, width, height);
        Ok(CudaImage { ctx, buf, layout })
    }

    /// Reference to the underlying CUDA device (shared from the [`StreamContext`]).
    #[cfg(not(tarpaulin_include))]
    pub fn device(&self) -> &Arc<CudaContext> {
        self.ctx.device()
    }

    /// Create a GPU image from host data.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The stream context providing the CUDA device for allocation
    /// * `channels` - Number of channels per pixel
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - Host data buffer (must have length == width * height * channels)
    ///
    /// # Errors
    ///
    /// Returns `NppError::InvalidArgument` if data length does not match dimensions.
    /// Returns `NppError::Cuda` if device allocation or copy fails.
    #[cfg(not(tarpaulin_include))]
    pub fn from_host(
        ctx: Arc<StreamContext>,
        channels: u8,
        width: u32,
        height: u32,
        data: &[T],
    ) -> Result<Self, NppError> {
        validate_dims(channels, width, height)?;
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
        let buf = ctx.stream().clone_htod(data)?;
        let layout = CudaLayout::row_major_packed(channels, width, height);
        Ok(CudaImage { ctx, buf, layout })
    }

    /// Get the image dimensions as (channels, width, height).
    #[cfg(not(tarpaulin_include))]
    pub fn dimensions(&self) -> (u8, u32, u32) {
        self.layout.dimensions()
    }

    /// Get the image width in pixels.
    #[cfg(not(tarpaulin_include))]
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the image height in pixels.
    #[cfg(not(tarpaulin_include))]
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// Get the number of channels per pixel.
    #[cfg(not(tarpaulin_include))]
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Compute the linear index for a pixel at (x, y).
    #[cfg(not(tarpaulin_include))]
    pub fn get_index(&self, x: u32, y: u32) -> usize {
        (y as usize) * self.layout.height_stride + (x as usize) * self.layout.width_stride
    }

    /// Get the (x, y) coordinates of the first pixel in the image.
    #[cfg(not(tarpaulin_include))]
    pub fn get_start_point(&self) -> (u32, u32) {
        let img_index = self.layout.img_index;
        let x = (img_index / (self.layout.channels as usize)) as u32;
        let y = 0;
        (x, y)
    }

    /// Get the bounding box of the image as (x_min, y_min, x_max, y_max).
    #[cfg(not(tarpaulin_include))]
    pub fn bounds(&self) -> (u32, u32, u32, u32) {
        (0, 0, self.layout.width, self.layout.height)
    }

    /// Check if a pixel coordinate (x, y) is within the image bounds.
    #[cfg(not(tarpaulin_include))]
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
    #[cfg(not(tarpaulin_include))]
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
        let start = self.layout.img_index + img_index;
        let len = (h as usize) * self.layout.height_stride;

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
            ctx: self.ctx.clone(),
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
    #[cfg(not(tarpaulin_include))]
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
        let start = self.layout.img_index + img_index;
        let len = (h as usize) * self.layout.height_stride;

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
            ctx: self.ctx.clone(),
            view,
            layout,
        })
    }
}

/// A borrowed read-only view of a GPU image.
///
/// Fields and methods on this type are used by M2 sub-image operations;
/// unused in M1 but retained for API stability.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CudaImageView<'a, T: NppPixelType> {
    pub(crate) ctx: Arc<StreamContext>,
    pub(crate) view: CudaView<'a, T>,
    pub(crate) layout: CudaLayout,
}

impl<'a, T: NppPixelType> CudaImageView<'a, T> {
    /// Get the image width in pixels.
    #[cfg(not(tarpaulin_include))]
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the image height in pixels.
    #[cfg(not(tarpaulin_include))]
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// Get the number of channels per pixel.
    #[cfg(not(tarpaulin_include))]
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Get the image dimensions as (channels, width, height).
    #[cfg(not(tarpaulin_include))]
    pub fn dimensions(&self) -> (u8, u32, u32) {
        self.layout.dimensions()
    }

    /// Get the (x, y) coordinates of the first pixel in the view.
    #[cfg(not(tarpaulin_include))]
    pub fn get_start_point(&self) -> (u32, u32) {
        let img_index = self.layout.img_index;
        let x = (img_index / (self.layout.channels as usize)) as u32;
        let y = 0;
        (x, y)
    }

    /// Extract the raw device pointer for NPP calls.
    ///
    /// See `docs/spike-cudarc-ptr-bridge.md` for the authoritative pattern.
    ///
    /// # SyncOnDrop guard lifetime
    ///
    /// The `SyncOnDrop` guard returned by `device_ptr(stream)` lives within this
    /// function scope. The caller must ensure the resulting `*const T` is used
    /// before the guard is dropped (i.e., pass it immediately to an NPP FFI call
    /// in the same enclosing scope).
    #[allow(dead_code)]
    #[cfg(not(tarpaulin_include))]
    pub(crate) fn device_ptr(&self) -> *const T {
        let stream = self.ctx.stream();
        let (cu_ptr, _guard) = cudarc::driver::DevicePtr::device_ptr(&self.view, stream);
        cu_ptr as *const T
    }
}

/// A borrowed mutable view of a GPU image.
///
/// Fields and methods on this type are used by M2 sub-image operations;
/// unused in M1 but retained for API stability.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CudaImageViewMut<'a, T: NppPixelType> {
    pub(crate) ctx: Arc<StreamContext>,
    pub(crate) view: CudaViewMut<'a, T>,
    pub(crate) layout: CudaLayout,
}

impl<'a, T: NppPixelType> CudaImageViewMut<'a, T> {
    /// Get the image width in pixels.
    #[cfg(not(tarpaulin_include))]
    pub fn width(&self) -> u32 {
        self.layout.width
    }

    /// Get the image height in pixels.
    #[cfg(not(tarpaulin_include))]
    pub fn height(&self) -> u32 {
        self.layout.height
    }

    /// Get the number of channels per pixel.
    #[cfg(not(tarpaulin_include))]
    pub fn channels(&self) -> u8 {
        self.layout.channels
    }

    /// Get the image dimensions as (channels, width, height).
    #[cfg(not(tarpaulin_include))]
    pub fn dimensions(&self) -> (u8, u32, u32) {
        self.layout.dimensions()
    }

    /// Get the (x, y) coordinates of the first pixel in the view.
    #[cfg(not(tarpaulin_include))]
    pub fn get_start_point(&self) -> (u32, u32) {
        let img_index = self.layout.img_index;
        let x = (img_index / (self.layout.channels as usize)) as u32;
        let y = 0;
        (x, y)
    }

    /// Extract the raw mutable device pointer for NPP calls.
    ///
    /// See `docs/spike-cudarc-ptr-bridge.md` for the authoritative pattern.
    ///
    /// # SyncOnDrop guard lifetime
    ///
    /// The `SyncOnDrop` guard returned by `device_ptr_mut(stream)` lives within
    /// this function scope. The caller must ensure the resulting `*mut T` is
    /// used before the guard is dropped (i.e., pass it immediately to an NPP
    /// FFI call in the same enclosing scope).
    #[allow(dead_code)]
    #[cfg(not(tarpaulin_include))]
    pub(crate) fn device_ptr_mut(&mut self) -> *mut T {
        let stream = self.ctx.stream();
        let (cu_ptr, _guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut self.view, stream);
        cu_ptr as *mut T
    }
}

/// Copy a GPU image to host memory.
///
/// # Ordering contract
///
/// NPP `_Ctx` operations are enqueued on the forked stream. This readback
/// performs:
///
/// 1. A host-blocking [`StreamContext::synchronize()`] on the forked stream
///    (the "fence"), ensuring all prior `_Ctx` work is complete.
/// 2. A per-stream DtoH copy via `stream.clone_dtoh` (cudarc 0.19.x).
///
/// The fence in step 1 makes step 2 safe. Without it, the stream copy
/// could race with the forked-stream NPP work.
///
/// See [`docs/stream-context.md`](https://github.com/vansweej/npp-rs/blob/main/docs/stream-context.md)
/// for the full rationale.
#[cfg(not(tarpaulin_include))]
impl<T: NppPixelType> TryFrom<&CudaImage<T>> for Vec<T> {
    type Error = NppError;

    fn try_from(img: &CudaImage<T>) -> Result<Self, Self::Error> {
        // Step 1: host fence — block until all forked-stream work is done
        img.ctx.synchronize()?;
        // Step 2: per-stream DtoH copy via CudaStream::clone_dtoh
        let host: Vec<T> = img.ctx.stream().clone_dtoh(&img.buf)?;
        Ok(host)
    }
}

/// Copy a borrowed image view to host memory.
///
/// # Ordering contract
///
/// Identical to the owned `TryFrom<&CudaImage<T>>` path: a host-blocking
/// [`StreamContext::synchronize`] fence, then a per-stream DtoH copy via
/// `stream.clone_dtoh`. The fence makes the stream copy safe against
/// forked-stream NPP work.
///
/// # Strided result
///
/// The returned `Vec` spans `height * parent_height_stride` elements — for a
/// sub-image narrower than its parent, it includes inter-row parent pixels
/// between ROI rows. Callers needing a packed ROI must de-stride, or read an
/// owned destination instead.
#[cfg(not(tarpaulin_include))]
impl<'a, T: NppPixelType> TryFrom<&CudaImageView<'a, T>> for Vec<T> {
    type Error = NppError;

    fn try_from(v: &CudaImageView<'a, T>) -> Result<Self, Self::Error> {
        v.ctx.synchronize()?;
        // Per-stream DtoH copy via CudaStream::clone_dtoh.
        // clone_dtoh returns a new Vec<T>, replacing the manual zeroed-buffer pattern.
        let host: Vec<T> = v.ctx.stream().clone_dtoh(&v.view)?;
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_dims_rejects_zero_channels() {
        assert!(matches!(
            validate_dims(0, 4, 4),
            Err(NppError::InvalidArgument(_))
        ));
    }

    #[test]
    fn validate_dims_rejects_zero_width() {
        assert!(matches!(
            validate_dims(3, 0, 4),
            Err(NppError::InvalidArgument(_))
        ));
    }

    #[test]
    fn validate_dims_rejects_zero_height() {
        assert!(matches!(
            validate_dims(3, 4, 0),
            Err(NppError::InvalidArgument(_))
        ));
    }

    #[test]
    fn validate_dims_accepts_nonzero() {
        assert!(validate_dims(3, 640, 480).is_ok());
    }
}
