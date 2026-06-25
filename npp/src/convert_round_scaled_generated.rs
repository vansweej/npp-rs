//! GENERATED — re-run `cargo run --example gen_convert_round_scaled_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{ConvertRoundedScaled, RoundMode};
use crate::impl_convert_rounded_scaled_for;
use npp_sys::NppiSize;

impl_convert_rounded_scaled_for!(i16, i8, "16s", "8s", {
        1 => npp_sys::nppiConvert_16s8s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u16, i16, "16u", "16s", {
        1 => npp_sys::nppiConvert_16u16s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u16, i8, "16u", "8s", {
        1 => npp_sys::nppiConvert_16u8s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(f32, i16, "32f", "16s", {
        1 => npp_sys::nppiConvert_32f16s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(f32, u16, "32f", "16u", {
        1 => npp_sys::nppiConvert_32f16u_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(f32, i32, "32f", "32s", {
        1 => npp_sys::nppiConvert_32f32s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(f32, u32, "32f", "32u", {
        1 => npp_sys::nppiConvert_32f32u_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(f32, i8, "32f", "8s", {
        1 => npp_sys::nppiConvert_32f8s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(f32, u8, "32f", "8u", {
        1 => npp_sys::nppiConvert_32f8u_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(i32, i16, "32s", "16s", {
        1 => npp_sys::nppiConvert_32s16s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(i32, u16, "32s", "16u", {
        1 => npp_sys::nppiConvert_32s16u_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u32, i16, "32u", "16s", {
        1 => npp_sys::nppiConvert_32u16s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u32, u16, "32u", "16u", {
        1 => npp_sys::nppiConvert_32u16u_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u32, i32, "32u", "32s", {
        1 => npp_sys::nppiConvert_32u32s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u32, i8, "32u", "8s", {
        1 => npp_sys::nppiConvert_32u8s_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u32, u8, "32u", "8u", {
        1 => npp_sys::nppiConvert_32u8u_C1RSfs_Ctx,
});

impl_convert_rounded_scaled_for!(u8, i8, "8u", "8s", {
        1 => npp_sys::nppiConvert_8u8s_C1RSfs_Ctx,
});
