//! GENERATED — re-run `cargo run --example gen_resize_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{Resize, ResizeInterpolation};
use crate::impl_resize_for;
use npp_sys::{NppiRect, NppiSize};

impl_resize_for!(i16, "16s", {
        1 => npp_sys::nppiResize_16s_C1R,
        3 => npp_sys::nppiResize_16s_C3R,
        4 => npp_sys::nppiResize_16s_C4R,
});

impl_resize_for!(u16, "16u", {
        1 => npp_sys::nppiResize_16u_C1R,
        3 => npp_sys::nppiResize_16u_C3R,
        4 => npp_sys::nppiResize_16u_C4R,
});

impl_resize_for!(f32, "32f", {
        1 => npp_sys::nppiResize_32f_C1R,
        3 => npp_sys::nppiResize_32f_C3R,
        4 => npp_sys::nppiResize_32f_C4R,
});

impl_resize_for!(u8, "8u", {
        1 => npp_sys::nppiResize_8u_C1R,
        3 => npp_sys::nppiResize_8u_C3R,
        4 => npp_sys::nppiResize_8u_C4R,
});
