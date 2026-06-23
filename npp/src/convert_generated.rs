//! GENERATED — re-run `cargo run --example gen_convert_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::ConvertTo;
use crate::impl_convert_for;
use npp_sys::NppiSize;

impl_convert_for!(u8, f32, "8u", "32f", {
        1 => npp_sys::nppiConvert_8u32f_C1R_Ctx,
        3 => npp_sys::nppiConvert_8u32f_C3R_Ctx,
});
