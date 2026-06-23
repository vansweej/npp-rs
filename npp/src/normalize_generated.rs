//! GENERATED — re-run `cargo run --example gen_normalize_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{ConvertTo, Normalize};
use crate::impl_normalize_for;

impl_normalize_for!(i16, 32767.0_f32, "16s");
impl_normalize_for!(u16, 65535.0_f32, "16u");
impl_normalize_for!(u8, 255.0_f32, "8u");
