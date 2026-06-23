//! Hand-written cross-type pixel format operation: `Normalize`.
//!
//! `ConvertTo` is generated in `convert_generated.rs` via the
//! `impl_convert_for!` macro. This module now hosts only the hand-written
//! `Normalize` slice. Generalizing Normalize across the alphabet is
//! **deferred to F5.2**.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{ConvertTo, Normalize};
use cudarc::driver::DevicePtrMut;
use npp_sys::NppiSize;
use std::mem::size_of;

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
///
/// `Normalize` uses the generated `ConvertTo<f32> for CudaImage<u8>` from
/// `convert_generated.rs` (behaviourally identical to the removed hand-written impl).
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
