//! In-crate ROI golden test for Resize via the `resize_into_8u` engine and a
//! `CudaImageView` source.
//!
//! ## Why in-crate, not in `npp/tests/`
//!
//! This test exercises `pub(crate)` items — the ROI engine function
//! `crate::resize_generated::resize_into_8u` and the `CudaImageView::device_ptr()` accessor —
//! that external integration tests (compiling as a separate crate) cannot reach.
//! **Do not "fix" by moving to `npp/tests/`.**
//!
//! ## Manual GPU pin procedure
//!
//! 1. Run: `nix develop . --command cargo test --features gpu resize_roi_tests`
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

use crate::image::CudaImage;
use crate::imageops::ResizeInterpolation;
use crate::stream::stream_context_for;
use crate::test_helpers::assert_golden;
use std::convert::TryFrom;

/// Golden output for Resize ROI u8 C3 NearestNeighbor 12×4 → 6×2.
/// Generated on NVIDIA GPU via `resize_into_8u` with `CudaImageView` source.
const EXPECTED: &[u8] = &[
    0, 64, 128, 42, 64, 128, 84, 64, 128, 126, 64, 128, 168, 64, 128, 210, 64, 128, 0, 128, 128,
    42, 128, 128, 84, 128, 128, 126, 128, 128, 168, 128, 128, 210, 128, 128,
];

#[test]
fn test_resize_roi_u8_c3_nn() {
    let ctx = stream_context_for(0).expect("stream context");

    // Parent: 3-channel u8, 12x8.
    let mut data = Vec::with_capacity((12 * 8 * 3) as usize);
    for y in 0..8u32 {
        for x in 0..12u32 {
            data.push((x * 21) as u8);
            data.push((y * 32) as u8);
            data.push(128u8);
        }
    }
    let parent = CudaImage::from_host(ctx.clone(), 3, 12, 8, &data).expect("parent allocation");

    // Source view: y-offset 2, height 4, full width (contiguous source read).
    let view = parent.sub_image(0, 2, 12, 4).expect("sub-image");

    // Owned destination: 6x2 (downscale 12x4 → 6x2).
    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, 6, 2).expect("dst allocation");

    // ── Route pointer extraction through accessor methods ──
    let src_ptr = view.device_ptr();
    let src_step_bytes = (view.layout.height_stride * std::mem::size_of::<u8>()) as i32;
    let dst_step_bytes = (dst.layout.height_stride * std::mem::size_of::<u8>()) as i32;
    // Precompute dst values before device_ptr_mut (avoids E0502 — SyncOnDrop
    // keeps &mut dst.buf alive across drop scope).
    let dst_w = dst.width();
    let dst_h = dst.height();
    let raw_ctx = view.ctx.raw_ctx();

    // ── FFI call: device_ptr_mut guard scoped to this block ──
    {
        let (dst_cu_ptr, _dst_guard) =
            cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, dst.ctx.stream());
        let dst_ptr = dst_cu_ptr as *mut u8;
        crate::resize_generated::resize_into_8u(
            src_ptr,
            src_step_bytes,
            view.width(),
            view.height(),
            view.channels(),
            dst_ptr,
            dst_step_bytes,
            dst_w,
            dst_h,
            ResizeInterpolation::NearestNeighbor,
            raw_ctx,
        )
        .expect("resize_into_8u ROI");
    }

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "resize ROI u8 C3 NN");
}
