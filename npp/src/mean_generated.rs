//! GENERATED — re-run `cargo run --example gen_mean_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::Mean;
use crate::impl_mean_for;
use npp_sys::NppiSize;

impl_mean_for!(i16, "16s", {
        1 => (npp_sys::nppiMean_16s_C1R, npp_sys::nppiMeanGetBufferHostSize_16s_C1R),
        3 => (npp_sys::nppiMean_16s_C3R, npp_sys::nppiMeanGetBufferHostSize_16s_C3R),
        4 => (npp_sys::nppiMean_16s_C4R, npp_sys::nppiMeanGetBufferHostSize_16s_C4R),
});

impl_mean_for!(u16, "16u", {
        1 => (npp_sys::nppiMean_16u_C1R, npp_sys::nppiMeanGetBufferHostSize_16u_C1R),
        3 => (npp_sys::nppiMean_16u_C3R, npp_sys::nppiMeanGetBufferHostSize_16u_C3R),
        4 => (npp_sys::nppiMean_16u_C4R, npp_sys::nppiMeanGetBufferHostSize_16u_C4R),
});

impl_mean_for!(f32, "32f", {
        1 => (npp_sys::nppiMean_32f_C1R, npp_sys::nppiMeanGetBufferHostSize_32f_C1R),
        3 => (npp_sys::nppiMean_32f_C3R, npp_sys::nppiMeanGetBufferHostSize_32f_C3R),
        4 => (npp_sys::nppiMean_32f_C4R, npp_sys::nppiMeanGetBufferHostSize_32f_C4R),
});

impl_mean_for!(u8, "8u", {
        1 => (npp_sys::nppiMean_8u_C1R, npp_sys::nppiMeanGetBufferHostSize_8u_C1R),
        3 => (npp_sys::nppiMean_8u_C3R, npp_sys::nppiMeanGetBufferHostSize_8u_C3R),
        4 => (npp_sys::nppiMean_8u_C4R, npp_sys::nppiMeanGetBufferHostSize_8u_C4R),
});
