//! GENERATED — re-run `cargo run --example gen_convert_round_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{ConvertRounded, RoundMode};
use crate::impl_convert_rounded_for;
use npp_sys::NppiSize;

impl_convert_rounded_for!(f32, i16, "32f", "16s", {
        1 => npp_sys::nppiConvert_32f16s_C1R_Ctx,
        3 => npp_sys::nppiConvert_32f16s_C3R_Ctx,
        4 => npp_sys::nppiConvert_32f16s_C4R_Ctx,
});

impl_convert_rounded_for!(f32, u16, "32f", "16u", {
        1 => npp_sys::nppiConvert_32f16u_C1R_Ctx,
        3 => npp_sys::nppiConvert_32f16u_C3R_Ctx,
        4 => npp_sys::nppiConvert_32f16u_C4R_Ctx,
});

impl_convert_rounded_for!(f32, i8, "32f", "8s", {
        1 => npp_sys::nppiConvert_32f8s_C1R_Ctx,
        3 => npp_sys::nppiConvert_32f8s_C3R_Ctx,
        4 => npp_sys::nppiConvert_32f8s_C4R_Ctx,
});

impl_convert_rounded_for!(f32, u8, "32f", "8u", {
        1 => npp_sys::nppiConvert_32f8u_C1R_Ctx,
        3 => npp_sys::nppiConvert_32f8u_C3R_Ctx,
        4 => npp_sys::nppiConvert_32f8u_C4R_Ctx,
});
