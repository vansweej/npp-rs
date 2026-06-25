//! In-crate ROI golden test for SwapChannels via the `swap_into_8u` engine and
//! a `CudaImageView` source.
//!
//! ## Why in-crate, not in `npp/tests/`
//!
//! This test exercises `pub(crate)` items — the ROI engine function
//! `crate::swap_channels_generated::swap_into_8u` and the `CudaImageView::device_ptr()` accessor — that
//! external integration tests (compiling as a separate crate) cannot reach.
//! **Do not "fix" by moving to `npp/tests/`.**
//!
//! ## Manual GPU pin procedure
//!
//! 1. Run: `nix develop . --command cargo test --features gpu swap_channels_roi_tests`
//! 2. It will print captured bytes and panic ("golden reference not yet pinned").
//! 3. Copy the printed byte literal into `EXPECTED` below.
//! 4. Re-run to confirm.
//!
//! ## Readback note
//!
//! This test reads an **owned** destination (`CudaImage::<u8>`), producing a
//! contiguous host `Vec`. This avoids the strided readback gap that a direct
//! `TryFrom<&CudaImageView>` would produce (the view readback spans `h *
//! parent_height_stride` elements, including inter-row padding).

use crate::cuda::default_cuda_device;
use crate::image::CudaImage;
use crate::stream::stream_context_for;
use crate::test_helpers::assert_golden;
use std::convert::TryFrom;

/// Golden output for SwapChannels ROI u8 BGRA→RGB 12×4.
/// Generated on NVIDIA GPU via `swap_into_8u` with `CudaImageView` source.
const EXPECTED: &[u8] = &[
    128, 64, 0, 128, 64, 21, 128, 64, 42, 128, 64, 63, 128, 64, 84, 128, 64, 105, 128, 64, 126,
    128, 64, 147, 128, 64, 168, 128, 64, 189, 128, 64, 210, 128, 64, 231, 128, 96, 0, 128, 96, 21,
    128, 96, 42, 128, 96, 63, 128, 96, 84, 128, 96, 105, 128, 96, 126, 128, 96, 147, 128, 96, 168,
    128, 96, 189, 128, 96, 210, 128, 96, 231, 128, 128, 0, 128, 128, 21, 128, 128, 42, 128, 128,
    63, 128, 128, 84, 128, 128, 105, 128, 128, 126, 128, 128, 147, 128, 128, 168, 128, 128, 189,
    128, 128, 210, 128, 128, 231, 128, 160, 0, 128, 160, 21, 128, 160, 42, 128, 160, 63, 128, 160,
    84, 128, 160, 105, 128, 160, 126, 128, 160, 147, 128, 160, 168, 128, 160, 189, 128, 160, 210,
    128, 160, 231,
];

#[test]
fn test_swap_channels_roi_u8_bgra_to_rgb() {
    let _dev = default_cuda_device().expect("CUDA device init");
    let ctx = stream_context_for(0).expect("stream context");

    // Parent: 4-channel u8 BGRA, 12x8.
    let mut data = Vec::with_capacity((12 * 8 * 4) as usize);
    for y in 0..8u32 {
        for x in 0..12u32 {
            data.push((x * 21) as u8); // B
            data.push((y * 32) as u8); // G
            data.push(128u8); // R
            data.push(255u8); // A
        }
    }
    let parent = CudaImage::from_host(ctx.clone(), 4, 12, 8, &data).expect("parent allocation");

    // Source view: y-offset 2, height 4, full width.
    let view = parent.sub_image(0, 2, 12, 4).expect("sub-image");

    // Owned destination: 3-channel RGB, same dimensions.
    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, 12, 4).expect("dst allocation");

    // ── Route pointer extraction through accessor methods ──
    let src_ptr = view.device_ptr();
    let src_step_bytes = (view.layout.height_stride * std::mem::size_of::<u8>()) as i32;
    let dst_ptr = *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut u8;
    let dst_step_bytes = (dst.layout.height_stride * std::mem::size_of::<u8>()) as i32;

    crate::swap_channels_generated::swap_into_8u(
        src_ptr,
        src_step_bytes,
        view.width(),
        view.height(),
        view.channels(),
        dst_ptr,
        dst_step_bytes,
        dst.width(),
        dst.height(),
        view.ctx.raw_ctx(),
    )
    .expect("swap_into_8u ROI");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "swap_channels ROI u8 BGRA->RGB");
}
