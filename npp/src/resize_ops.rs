use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{Resize, ResizeInterpolation};
use npp_sys::{
    nppiResize_32f_C3R, nppiResize_8u_C3R, NppiInterpolationMode_NPPI_INTER_CUBIC,
    NppiInterpolationMode_NPPI_INTER_LANCZOS, NppiInterpolationMode_NPPI_INTER_LINEAR,
    NppiInterpolationMode_NPPI_INTER_NN, NppiInterpolationMode_NPPI_INTER_SUPER, NppiRect,
    NppiSize,
};

fn interpolation_mode(inter: ResizeInterpolation) -> i32 {
    match inter {
        ResizeInterpolation::NearestNeighbor => NppiInterpolationMode_NPPI_INTER_NN as i32,
        ResizeInterpolation::Linear => NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
        ResizeInterpolation::Cubic => NppiInterpolationMode_NPPI_INTER_CUBIC as i32,
        ResizeInterpolation::Super => NppiInterpolationMode_NPPI_INTER_SUPER as i32,
        ResizeInterpolation::Lanczos => NppiInterpolationMode_NPPI_INTER_LANCZOS as i32,
    }
}

/// Resize for `CudaImage<u8>` over `nppiResize_8u_C3R` (3-channel packed RGB).
///
/// The raw pointer for both src and dst is offset by `layout.img_index` so
/// this impl works correctly on sub-images created via `CudaImage::sub_image`
/// (whose `layout.img_index` carries the parent's offset). Because the
/// `CudaImage` stores a full `CudaSlice<T>` whose base is the allocation
/// origin, and `img_index` is the element offset to the first pixel of the
/// (possibly sub-)image, we apply the offset uniformly — it is zero for
/// full-size images.
///
/// # Precondition
///
/// `self` and `dst` must refer to **non-overlapping** device buffers.
/// Passing overlapping ROIs is undefined behaviour in NPP (C4).
impl Resize for CudaImage<u8> {
    fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError> {
        let (src_w, src_h) = (self.width(), self.height());
        let (dst_w, dst_h) = (dst.width(), dst.height());

        let src_size = NppiSize {
            width: src_w as i32,
            height: src_h as i32,
        };
        let dst_size = NppiSize {
            width: dst_w as i32,
            height: dst_h as i32,
        };
        let src_rect = NppiRect {
            x: 0,
            y: 0,
            width: src_w as i32,
            height: src_h as i32,
        };
        let dst_rect = NppiRect {
            x: 0,
            y: 0,
            width: dst_w as i32,
            height: dst_h as i32,
        };

        // ── Raw pointers via DevicePtr/DevicePtrMut trait ──────────────
        // Both the const and mutable CUdeviceptrs are offset by img_index
        // so sub-images (which have img_index != 0) address the correct
        // device memory. The height_stride is inherited from the parent
        // layout and is already correct.
        let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
        let src_ptr = (src_base + self.layout.img_index as u64) as *const u8;
        let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (*dst_base + dst.layout.img_index as u64) as *mut u8;

        let status = unsafe {
            nppiResize_8u_C3R(
                src_ptr,
                self.layout.height_stride as i32,
                src_size,
                src_rect,
                dst_ptr,
                dst.layout.height_stride as i32,
                dst_size,
                dst_rect,
                interpolation_mode(inter),
            )
        };
        check_status(status)
    }
}

/// Resize for `CudaImage<f32>` over `nppiResize_32f_C3R` (3-channel packed float).
///
/// # nStep unit conversion
///
/// NPP's `nStep` is in **bytes**. `layout.height_stride` stores the per-row
/// element count; we multiply by `size_of::<f32>()` to produce the byte step.
/// This is the correct pattern for all non-`u8` types — for `u8` the two
/// coincide, but for wider types the explicit conversion prevents the C11
/// class of bug.
///
/// # Precondition
///
/// `self` and `dst` must refer to **non-overlapping** device buffers.
impl Resize for CudaImage<f32> {
    fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError> {
        let (src_w, src_h) = (self.width(), self.height());
        let (dst_w, dst_h) = (dst.width(), dst.height());

        let src_size = NppiSize {
            width: src_w as i32,
            height: src_h as i32,
        };
        let dst_size = NppiSize {
            width: dst_w as i32,
            height: dst_h as i32,
        };
        let src_rect = NppiRect {
            x: 0,
            y: 0,
            width: src_w as i32,
            height: src_h as i32,
        };
        let dst_rect = NppiRect {
            x: 0,
            y: 0,
            width: dst_w as i32,
            height: dst_h as i32,
        };

        // nStep is in BYTES. height_stride counts f32 elements. Convert.
        let src_step_bytes = (self.layout.height_stride * std::mem::size_of::<f32>()) as i32;
        let dst_step_bytes = (dst.layout.height_stride * std::mem::size_of::<f32>()) as i32;

        // Raw pointer offset via DevicePtr/DevicePtrMut (handles sub-images).
        let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
        let src_ptr = (src_base + self.layout.img_index as u64) as *const f32;
        let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (*dst_base + dst.layout.img_index as u64) as *mut f32;

        let status = unsafe {
            nppiResize_32f_C3R(
                src_ptr,
                src_step_bytes,
                src_size,
                src_rect,
                dst_ptr,
                dst_step_bytes,
                dst_size,
                dst_rect,
                interpolation_mode(inter),
            )
        };
        check_status(status)
    }
}
