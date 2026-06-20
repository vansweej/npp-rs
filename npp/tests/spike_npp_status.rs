// Status-code taxonomy (pinned from CUDA 12.9 — update on CUDA bump):
//   NPP_SUCCESS (≥ 0)                                   → supported
//   NPP_INTERPOLATION_ERROR (-22)                        → mode unsupported
//   step/size error (-201, pinned from this spike)       → harness bug (image too small)
//   anything else                                        → FAIL LOUD (spike was incomplete)

#![cfg(feature = "gpu")]

use npp_rs::cuda::default_cuda_device;
use npp_sys::{
    nppiFree, nppiMalloc_32f_C3, nppiMalloc_8u_C3, nppiResize_32f_C3R, nppiResize_8u_C3R,
    NppStatus, NppiInterpolationMode_NPPI_INTER_LANCZOS, NppiRect, NppiSize,
};
use std::ffi::c_void;
use std::sync::OnceLock;

/// Retain a CUDA device handle for the lifetime of the process so the context
/// is alive when `nppiMalloc` calls `cudaMallocPitch`.
fn ensure_cuda() {
    static DEV: OnceLock<()> = OnceLock::new();
    DEV.get_or_init(|| {
        default_cuda_device().expect("CUDA device init for spike");
    });
}

/// Positive case: 32f Lanczos resize on a comfortable 64×64 → 32×32.
/// Expects status >= 0 (success).
const EXPECT_POSITIVE_MIN: i32 = 0;

/// Negative case: invalid interpolation mode value 999 on 8u resize.
/// Expects NPP_INTERPOLATION_ERROR (exact negative code, pinned from first run).
const EXPECT_NEGATIVE: i32 = -22;

/// Harness-bug case: 32f Lanczos on 1×1 → 1×1 (image too small).
/// Expects a step/size error code -201 (pinned from CUDA 12.9).
const EXPECT_HARNESS_BUG: i32 = -201;

fn call_resize_32f(src_w: i32, src_h: i32, dst_w: i32, dst_h: i32, inter: i32) -> NppStatus {
    ensure_cuda();
    let mut src_step: i32 = 0;
    let mut dst_step: i32 = 0;

    unsafe {
        let src = nppiMalloc_32f_C3(src_w, src_h, &mut src_step);
        assert!(!src.is_null(), "src allocation failed");
        let dst = nppiMalloc_32f_C3(dst_w, dst_h, &mut dst_step);
        assert!(!dst.is_null(), "dst allocation failed");

        let src_size = NppiSize {
            width: src_w,
            height: src_h,
        };
        let dst_size = NppiSize {
            width: dst_w,
            height: dst_h,
        };
        // NppiRect uses the same fields as NppiSize but semantically different.
        let src_roi = NppiRect {
            x: 0,
            y: 0,
            width: src_w,
            height: src_h,
        };
        let dst_roi = NppiRect {
            x: 0,
            y: 0,
            width: dst_w,
            height: dst_h,
        };

        let status = nppiResize_32f_C3R(
            src as *const _,
            src_step,
            src_size,
            src_roi,
            dst as *mut _,
            dst_step,
            dst_size,
            dst_roi,
            inter,
        );

        nppiFree(src as *mut c_void);
        nppiFree(dst as *mut c_void);

        status
    }
}

fn call_resize_8u(src_w: i32, src_h: i32, dst_w: i32, dst_h: i32, inter: i32) -> NppStatus {
    ensure_cuda();
    let mut src_step: i32 = 0;
    let mut dst_step: i32 = 0;

    unsafe {
        let src = nppiMalloc_8u_C3(src_w, src_h, &mut src_step);
        assert!(!src.is_null(), "src allocation failed");
        let dst = nppiMalloc_8u_C3(dst_w, dst_h, &mut dst_step);
        assert!(!dst.is_null(), "dst allocation failed");

        let src_size = NppiSize {
            width: src_w,
            height: src_h,
        };
        let dst_size = NppiSize {
            width: dst_w,
            height: dst_h,
        };
        let src_roi = NppiRect {
            x: 0,
            y: 0,
            width: src_w,
            height: src_h,
        };
        let dst_roi = NppiRect {
            x: 0,
            y: 0,
            width: dst_w,
            height: dst_h,
        };

        let status = nppiResize_8u_C3R(
            src as *const _,
            src_step,
            src_size,
            src_roi,
            dst as *mut _,
            dst_step,
            dst_size,
            dst_roi,
            inter,
        );

        nppiFree(src as *mut c_void);
        nppiFree(dst as *mut c_void);

        status
    }
}

#[test]
fn spike_positive() {
    let status = call_resize_32f(
        64,
        64,
        32,
        32,
        NppiInterpolationMode_NPPI_INTER_LANCZOS as i32,
    );
    eprintln!("SPIKE POSITIVE: status = {status} (expected >= {EXPECT_POSITIVE_MIN})");
    assert!(
        status >= EXPECT_POSITIVE_MIN,
        "positive case should succeed"
    );
}

#[test]
fn spike_negative() {
    // Invalid interpolation value — guaranteed unsupported
    let status = call_resize_8u(64, 64, 32, 32, 999);
    eprintln!("SPIKE NEGATIVE: status = {status} (expected == {EXPECT_NEGATIVE})");
    assert_eq!(
        status, EXPECT_NEGATIVE,
        "negative case should match pinned error code"
    );
}

#[test]
fn spike_harness_bug() {
    let status = call_resize_32f(1, 1, 1, 1, NppiInterpolationMode_NPPI_INTER_LANCZOS as i32);
    eprintln!("SPIKE HARNESS_BUG: status = {status} (expected == {EXPECT_HARNESS_BUG})");
    assert_eq!(
        status, EXPECT_HARNESS_BUG,
        "harness-bug case should match pinned error code"
    );
}
