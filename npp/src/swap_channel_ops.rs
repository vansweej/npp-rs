use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::SwapChannels;
use npp_sys::{nppiSwapChannels_8u_C4C3R, NppiSize};
use std::os::raw::c_int;

/// BGRA→RGB channel reorder for `CudaImage<u8>` over `nppiSwapChannels_8u_C4C3R`.
///
/// Takes a 4-channel (BGRA) source and writes a 3-channel (RGB) destination.
/// The raw pointer offset logic mirrors the `Resize` impl — sub-images are
/// handled correctly via `layout.img_index`.
///
/// # Precondition
///
/// - `src` and `dst` must refer to **non-overlapping** device buffers.
/// - `src` must have 4 channels (BGRA), `dst` must have 3 channels (RGB).
/// - Dimensions (width, height) of src and dst must match (enforced at
///   runtime via `InvalidArgument` — survives `--release`).
///
/// # Errors
///
/// Returns `InvalidArgument` if src and dst dimensions disagree. Returns
/// `Npp` (containing the raw NPP status code) if the NPP call fails.
impl SwapChannels for CudaImage<u8> {
    fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError> {
        // Dimension agreement check (survives --release — C2 hardening)
        if self.width() != dst.width() || self.height() != dst.height() {
            return Err(NppError::InvalidArgument(
                "src and dst dimensions must match for bgra_to_rgb".into(),
            ));
        }

        let nppi_size = NppiSize {
            width: dst.width() as i32,
            height: dst.height() as i32,
        };

        let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
        let src_ptr = (src_base + self.layout.img_index as u64) as *const u8;
        let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (*dst_base + dst.layout.img_index as u64) as *mut u8;

        let order: [c_int; 3] = [2, 1, 0];
        let status = unsafe {
            nppiSwapChannels_8u_C4C3R(
                src_ptr,
                self.layout.height_stride as i32,
                dst_ptr,
                dst.layout.height_stride as i32,
                nppi_size,
                &order[0],
            )
        };
        check_status(status)
    }
}
