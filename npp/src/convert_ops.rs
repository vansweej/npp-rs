//! Cross-type pixel format conversion operations (hand-written).
//!
//! This module implements the `ConvertTo` and `Normalize` traits for supported pixel type pairs.
//! Currently: `u8 → f32` (C1, C3 channels only).

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{ConvertTo, Normalize};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use npp_sys::NppiSize;
use std::mem::size_of;

/// Implement `ConvertTo<f32>` for `CudaImage<u8>`.
///
/// Converts 8-bit unsigned integer pixels to 32-bit floating-point pixels.
/// Supported channel counts: C1 (1-channel), C3 (3-channel).
///
/// # Precondition
///
/// `self` and `dst` must not overlap in memory. This applies to
/// **neighbourhood-gather** operations; aliasing produces undefined results.
/// Purely **elementwise** operations may safely alias (see `Normalize`).
///
/// The destination image's `width`, `height`, and `channels` must match the source.
#[cfg(not(tarpaulin_include))]
impl ConvertTo<f32> for CudaImage<u8> {
    fn convert(&self, dst: &mut CudaImage<f32>) -> Result<(), NppError> {
        // Check 1: Agreement — dimensions and channels must match
        if self.width() != dst.width()
            || self.height() != dst.height()
            || self.channels() != dst.channels()
        {
            return Err(NppError::InvalidArgument(
                "src and dst dimensions/channels must match for convert".into(),
            ));
        }

        // Check 2: Support — only C1 and C3 are supported
        match self.channels() {
            1 | 3 => {}
            _ => {
                return Err(NppError::InvalidArgument(
                    "convert_8u32f only supports C1 and C3 channels".into(),
                ));
            }
        }

        // Extract raw pointers using the pointer-bridge pattern
        let cu_ptr = DevicePtr::device_ptr(&self.buf);
        let src_ptr = (cu_ptr + self.layout.img_index as u64) as *const u8;

        let cu_ptr_mut = *DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (cu_ptr_mut + dst.layout.img_index as u64) as *mut f32;

        // Calculate byte-steps
        let src_step = self.layout.height_stride as i32 * size_of::<u8>() as i32;
        let dst_step = dst.layout.height_stride as i32 * size_of::<f32>() as i32;

        // Create NppiSize from destination dimensions
        let size = NppiSize {
            width: dst.width() as i32,
            height: dst.height() as i32,
        };

        // Call the appropriate NPP function based on channel count
        let status = unsafe {
            match self.channels() {
                1 => npp_sys::nppiConvert_8u32f_C1R_Ctx(
                    src_ptr,
                    src_step,
                    dst_ptr,
                    dst_step,
                    size,
                    self.ctx.raw_ctx(),
                ),
                3 => npp_sys::nppiConvert_8u32f_C3R_Ctx(
                    src_ptr,
                    src_step,
                    dst_ptr,
                    dst_step,
                    size,
                    self.ctx.raw_ctx(),
                ),
                _ => unreachable!(), // Already checked above
            }
        };

        check_status(status)?;
        Ok(())
    }
}

/// Implement `Normalize<f32>` for `CudaImage<u8>`.
///
/// Normalizes 8-bit unsigned integer pixels to 32-bit floating-point pixels
/// in the range [0.0, 1.0]. Supported channel counts: C1 (1-channel), C3 (3-channel).
///
/// # Implementation
///
/// This is a two-step process:
/// 1. Convert `[0, 255]` to `[0.0, 255.0]` using `nppiConvert_8u32f_*_Ctx`
/// 2. Scale in-place by `1/255` using `nppiMulC_32f_*_Ctx`
///
/// The in-place scaling is safe because `MulC` is a purely elementwise operation
/// (output pixel (x,y) depends only on input pixel (x,y)), so reading and writing
/// the same buffer does not trigger the C4 gather-overlap hazard.
///
/// # Precondition
///
/// `self` and `dst` must not overlap in memory. The convert step obeys the non-overlap
/// requirement for **neighbourhood-gather** operations; the scale step is purely
/// **elementwise** and may safely alias.
///
/// The destination image's `width`, `height`, and `channels` must match the source.
#[cfg(not(tarpaulin_include))]
impl Normalize<f32> for CudaImage<u8> {
    fn normalize(&self, dst: &mut CudaImage<f32>) -> Result<(), NppError> {
        // Step 1: Convert [0, 255] to [0.0, 255.0]
        self.convert(dst)?;

        // Step 2: Scale in-place by 1/255 to get [0.0, 1.0]
        // MulC is elementwise — output (x,y) depends only on input (x,y) —
        // so reading/writing dst in place is sound and does not trigger the C4
        // gather-overlap hazard.

        // Extract mutable pointer for in-place scaling
        let cu_ptr_mut = *DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (cu_ptr_mut + dst.layout.img_index as u64) as *mut f32;

        // Calculate byte-step for f32
        let dst_step = dst.layout.height_stride as i32 * size_of::<f32>() as i32;

        // Create NppiSize from destination dimensions
        let size = NppiSize {
            width: dst.width() as i32,
            height: dst.height() as i32,
        };

        // Scale by 1/255 with channel-correct arity
        let status = unsafe {
            match dst.channels() {
                1 => {
                    // C1: scalar constant
                    let constant = 1.0_f32 / 255.0_f32;
                    npp_sys::nppiMulC_32f_C1R_Ctx(
                        dst_ptr,
                        dst_step,
                        constant,
                        dst_ptr,
                        dst_step,
                        size,
                        dst.ctx.raw_ctx(),
                    )
                }
                3 => {
                    // C3: array of 3 constants (one per channel)
                    let constants = [1.0_f32 / 255.0_f32; 3];
                    npp_sys::nppiMulC_32f_C3R_Ctx(
                        dst_ptr,
                        dst_step,
                        constants.as_ptr(),
                        dst_ptr,
                        dst_step,
                        size,
                        dst.ctx.raw_ctx(),
                    )
                }
                _ => unreachable!(), // Already checked in convert
            }
        };

        check_status(status)?;
        Ok(())
    }
}
