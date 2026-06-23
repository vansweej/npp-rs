//! GENERATED — re-run `cargo run --example gen_convert_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::ConvertTo;
use crate::impl_convert_for;
use npp_sys::NppiSize;

impl_convert_for!(i16, f32, "16s", "32f", {
        1 => npp_sys::nppiConvert_16s32f_C1R_Ctx,
        3 => npp_sys::nppiConvert_16s32f_C3R_Ctx,
        4 => npp_sys::nppiConvert_16s32f_C4R_Ctx,
});

impl_convert_for!(u16, f32, "16u", "32f", {
        1 => npp_sys::nppiConvert_16u32f_C1R_Ctx,
        3 => npp_sys::nppiConvert_16u32f_C3R_Ctx,
        4 => npp_sys::nppiConvert_16u32f_C4R_Ctx,
});

impl_convert_for!(u16, i32, "16u", "32s", {
        1 => npp_sys::nppiConvert_16u32s_C1R_Ctx,
        3 => npp_sys::nppiConvert_16u32s_C3R_Ctx,
        4 => npp_sys::nppiConvert_16u32s_C4R_Ctx,
});

impl_convert_for!(u8, i16, "8u", "16s", {
        1 => npp_sys::nppiConvert_8u16s_C1R_Ctx,
        3 => npp_sys::nppiConvert_8u16s_C3R_Ctx,
        4 => npp_sys::nppiConvert_8u16s_C4R_Ctx,
});

impl_convert_for!(u8, u16, "8u", "16u", {
        1 => npp_sys::nppiConvert_8u16u_C1R_Ctx,
        3 => npp_sys::nppiConvert_8u16u_C3R_Ctx,
        4 => npp_sys::nppiConvert_8u16u_C4R_Ctx,
});

impl_convert_for!(u8, f32, "8u", "32f", {
        1 => npp_sys::nppiConvert_8u32f_C1R_Ctx,
        3 => npp_sys::nppiConvert_8u32f_C3R_Ctx,
        4 => npp_sys::nppiConvert_8u32f_C4R_Ctx,
});

impl_convert_for!(u8, i32, "8u", "32s", {
        1 => npp_sys::nppiConvert_8u32s_C1R_Ctx,
        3 => npp_sys::nppiConvert_8u32s_C3R_Ctx,
        4 => npp_sys::nppiConvert_8u32s_C4R_Ctx,
});
