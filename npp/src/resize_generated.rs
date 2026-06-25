//! GENERATED — re-run `cargo run --example gen_resize_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{Resize, ResizeInterpolation};
use crate::impl_resize_for;
use npp_sys::{NppiRect, NppiSize};

impl_resize_for!(resize_into_16s, i16, "16s", {
        1 => npp_sys::nppiResize_16s_C1R_Ctx,
        3 => npp_sys::nppiResize_16s_C3R_Ctx,
        4 => npp_sys::nppiResize_16s_C4R_Ctx,
});

impl_resize_for!(resize_into_16u, u16, "16u", {
        1 => npp_sys::nppiResize_16u_C1R_Ctx,
        3 => npp_sys::nppiResize_16u_C3R_Ctx,
        4 => npp_sys::nppiResize_16u_C4R_Ctx,
});

impl_resize_for!(resize_into_32f, f32, "32f", {
        1 => npp_sys::nppiResize_32f_C1R_Ctx,
        3 => npp_sys::nppiResize_32f_C3R_Ctx,
        4 => npp_sys::nppiResize_32f_C4R_Ctx,
});

impl_resize_for!(resize_into_8u, u8, "8u", {
        1 => npp_sys::nppiResize_8u_C1R_Ctx,
        3 => npp_sys::nppiResize_8u_C3R_Ctx,
        4 => npp_sys::nppiResize_8u_C4R_Ctx,
});
