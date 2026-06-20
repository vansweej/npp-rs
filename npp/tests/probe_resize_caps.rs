//! GPU probe for the (type_token, interpolation) support matrix.
//!
//! Iterates over every NPP type that has a `nppiResize_<t>_C3R` symbol in the
//! corpus (skipping 16f), tests each `ResizeInterpolation` variant, and prints
//! a Rust array literal ready to paste into `npp/src/resize_caps.rs`.
//!
//! # Status-code taxonomy (pinned from CUDA 12.9, see spike_npp_status.rs):
//!
//! | Condition | Code | Meaning |
//! |-----------|------|---------|
//! | status >= 0 | success | supported pair |
//! | status == -22 | NPP_INTERPOLATION_ERROR | unsupported pair |
//! | status == -201 | step/size error | probe image too small (panic) |
//! | anything else | unknown | spike was incomplete (panic) |

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaDevice, DevicePtrMut};
use npp_rs::cuda::default_cuda_device;
use npp_rs::imageops::ResizeInterpolation;
use npp_sys::{
    nppiResize_16s_C3R, nppiResize_16u_C3R, nppiResize_32f_C3R, nppiResize_8u_C3R, NppStatus,
    NppiRect, NppiSize,
};
use std::sync::Arc;

const SRC_W: i32 = 64;
const SRC_H: i32 = 64;
const DST_W: i32 = 32;
const DST_H: i32 = 32;

/// Map ResizeInterpolation to the NPP integer enum value.
fn inter_mode(m: ResizeInterpolation) -> i32 {
    use npp_sys::{
        NppiInterpolationMode_NPPI_INTER_CUBIC, NppiInterpolationMode_NPPI_INTER_LANCZOS,
        NppiInterpolationMode_NPPI_INTER_LINEAR, NppiInterpolationMode_NPPI_INTER_NN,
        NppiInterpolationMode_NPPI_INTER_SUPER,
    };
    match m {
        ResizeInterpolation::NearestNeighbor => NppiInterpolationMode_NPPI_INTER_NN as i32,
        ResizeInterpolation::Linear => NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
        ResizeInterpolation::Cubic => NppiInterpolationMode_NPPI_INTER_CUBIC as i32,
        ResizeInterpolation::Super => NppiInterpolationMode_NPPI_INTER_SUPER as i32,
        ResizeInterpolation::Lanczos => NppiInterpolationMode_NPPI_INTER_LANCZOS as i32,
    }
}

/// Extract a raw mutable device pointer from a CudaSlice.
fn as_mut_ptr<T: cudarc::driver::DeviceRepr>(buf: &mut cudarc::driver::CudaSlice<T>) -> *mut T {
    let base = DevicePtrMut::device_ptr_mut(buf);
    *base as *mut T
}

/// Classify the NPP status and return true for supported, false for unsupported.
/// Panics on unexpected codes (means the spike was incomplete or the buffer is too small).
fn classify_status(type_token: &str, inter: ResizeInterpolation, status: NppStatus) -> bool {
    match status {
        s if s >= 0 => true,
        -22 => false,
        -201 => panic!("probe image too small for {type_token} {inter:?} (status={status})"),
        other => panic!("unknown status {other} for {type_token} {inter:?} — spike was incomplete"),
    }
}

/// Test a single (type_token, interpolation) pair.
/// Returns true if the pair is supported (status >= 0).
fn test_pair(dev: &Arc<CudaDevice>, type_token: &str, inter: ResizeInterpolation) -> bool {
    let (src_w, src_h) = (SRC_W as usize, SRC_H as usize);
    let (dst_w, dst_h) = (DST_W as usize, DST_H as usize);
    let ch = 3usize;

    let src_len = src_w * src_h * ch;
    let dst_len = dst_w * dst_h * ch;

    unsafe {
        match type_token {
            "8u" => {
                let mut src_buf = dev.alloc_zeros::<u8>(src_len).expect("alloc src");
                let mut dst_buf = dev.alloc_zeros::<u8>(dst_len).expect("alloc dst");

                let src_step = (SRC_W * ch as i32) as i32;
                let dst_step = (DST_W * ch as i32) as i32;

                let src_size = NppiSize {
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_size = NppiSize {
                    width: DST_W,
                    height: DST_H,
                };
                let src_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: DST_W,
                    height: DST_H,
                };

                let status = nppiResize_8u_C3R(
                    as_mut_ptr(&mut src_buf) as *const _,
                    src_step,
                    src_size,
                    src_roi,
                    as_mut_ptr(&mut dst_buf) as *mut _,
                    dst_step,
                    dst_size,
                    dst_roi,
                    inter_mode(inter),
                );
                classify_status(type_token, inter, status)
            }
            "16u" => {
                let mut src_buf = dev.alloc_zeros::<u16>(src_len).expect("alloc src");
                let mut dst_buf = dev.alloc_zeros::<u16>(dst_len).expect("alloc dst");

                let src_step = (SRC_W * ch as i32 * 2) as i32;
                let dst_step = (DST_W * ch as i32 * 2) as i32;

                let src_size = NppiSize {
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_size = NppiSize {
                    width: DST_W,
                    height: DST_H,
                };
                let src_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: DST_W,
                    height: DST_H,
                };

                let status = nppiResize_16u_C3R(
                    as_mut_ptr(&mut src_buf) as *const _,
                    src_step,
                    src_size,
                    src_roi,
                    as_mut_ptr(&mut dst_buf) as *mut _,
                    dst_step,
                    dst_size,
                    dst_roi,
                    inter_mode(inter),
                );
                classify_status(type_token, inter, status)
            }
            "16s" => {
                let mut src_buf = dev.alloc_zeros::<i16>(src_len).expect("alloc src");
                let mut dst_buf = dev.alloc_zeros::<i16>(dst_len).expect("alloc dst");

                let src_step = (SRC_W * ch as i32 * 2) as i32;
                let dst_step = (DST_W * ch as i32 * 2) as i32;

                let src_size = NppiSize {
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_size = NppiSize {
                    width: DST_W,
                    height: DST_H,
                };
                let src_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: DST_W,
                    height: DST_H,
                };

                let status = nppiResize_16s_C3R(
                    as_mut_ptr(&mut src_buf) as *const _,
                    src_step,
                    src_size,
                    src_roi,
                    as_mut_ptr(&mut dst_buf) as *mut _,
                    dst_step,
                    dst_size,
                    dst_roi,
                    inter_mode(inter),
                );
                classify_status(type_token, inter, status)
            }
            "32f" => {
                let mut src_buf = dev.alloc_zeros::<f32>(src_len).expect("alloc src");
                let mut dst_buf = dev.alloc_zeros::<f32>(dst_len).expect("alloc dst");

                let src_step = (SRC_W * ch as i32 * 4) as i32;
                let dst_step = (DST_W * ch as i32 * 4) as i32;

                let src_size = NppiSize {
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_size = NppiSize {
                    width: DST_W,
                    height: DST_H,
                };
                let src_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: SRC_W,
                    height: SRC_H,
                };
                let dst_roi = NppiRect {
                    x: 0,
                    y: 0,
                    width: DST_W,
                    height: DST_H,
                };

                let status = nppiResize_32f_C3R(
                    as_mut_ptr(&mut src_buf) as *const _,
                    src_step,
                    src_size,
                    src_roi,
                    as_mut_ptr(&mut dst_buf) as *mut _,
                    dst_step,
                    dst_size,
                    dst_roi,
                    inter_mode(inter),
                );
                classify_status(type_token, inter, status)
            }
            other => panic!("unexpected type_token without C3R symbol: {other}"),
        }
    }
}

#[test]
fn probe_resize_caps() {
    let device: Arc<CudaDevice> = default_cuda_device().expect("CUDA device init");

    // Types that have nppiResize_<t>_C3R symbols (from the corpus)
    let types = ["8u", "16u", "16s", "32f"];
    let modes = [
        ResizeInterpolation::NearestNeighbor,
        ResizeInterpolation::Linear,
        ResizeInterpolation::Cubic,
        ResizeInterpolation::Super,
        ResizeInterpolation::Lanczos,
    ];

    let mut supported: Vec<(&str, ResizeInterpolation)> = Vec::new();

    for t in &types {
        for m in &modes {
            if test_pair(&device, t, *m) {
                supported.push((*t, *m));
            }
        }
    }

    eprintln!("{:?}", supported);
}
