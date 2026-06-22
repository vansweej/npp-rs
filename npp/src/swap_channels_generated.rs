//! GENERATED — re-run `cargo run --example gen_swapchannels_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::SwapChannels;
use crate::impl_swap_channels_for;
use npp_sys::NppiSize;

impl_swap_channels_for!(i16, "16s", {
        4 => npp_sys::nppiSwapChannels_16s_C4C3R_Ctx,
});

impl_swap_channels_for!(u16, "16u", {
        4 => npp_sys::nppiSwapChannels_16u_C4C3R_Ctx,
});

impl_swap_channels_for!(f32, "32f", {
        4 => npp_sys::nppiSwapChannels_32f_C4C3R_Ctx,
});

impl_swap_channels_for!(i32, "32s", {
        4 => npp_sys::nppiSwapChannels_32s_C4C3R_Ctx,
});

impl_swap_channels_for!(u8, "8u", {
        4 => npp_sys::nppiSwapChannels_8u_C4C3R_Ctx,
});
