//! Safe Rust bindings for NVIDIA NPP image operations.
//!
//! Built on:
//! - `cudarc` for CUDA device management (`Arc<CudaDevice>`, `CudaSlice<T>`)
//! - `npp-sys` for generated FFI bindings to the NPP image domain (`nppi*` symbols)
//!
//! The core type is [`CudaImage<T>`](crate::image::CudaImage), where `T: NppPixelType` covers the full NPP
//! primitive alphabet (~9 types). Operation capability is expressed via traits
//! (e.g. [`Resize`](crate::imageops::Resize), [`SwapChannels`](crate::imageops::SwapChannels)): an unsupported `(type, op)` pair simply
//! has no trait impl, making it a compile-time error.
//!
//! Cross-type operations ([`ConvertTo`](crate::imageops::ConvertTo), [`Normalize`](crate::imageops::Normalize),
//! [`ConvertRounded`](crate::imageops::ConvertRounded),
//! [`ConvertRoundedScaled`](crate::imageops::ConvertRoundedScaled))
//! convert between different pixel types (e.g. `u8 → f32`). Normalize is generated
//! for integer→f32 pairs; `ConvertRounded` handles narrowing conversions with
//! explicit rounding mode (e.g. `f32 → u8`); `ConvertRoundedScaled` adds an
//! integer scaling factor (`C1RSfs` family).
//!
//! Round-trip to host memory uses `TryFrom<&CudaImage<T>> for Vec<T>`.
//! There is no `image` crate dependency in the core.
//!
//! # Test tiers
//!
//! - `cargo test` — pure-logic unit tests (no GPU required, no device init).
//! - `cargo test --features gpu` — device-dependent integration tests (requires
//!   an NVIDIA GPU + driver). This is a manual gate; CI has no GPU lane.
//!
//! # Safety
//!
//! The [`CudaImage`](crate::image::CudaImage) constructor requires an `Arc<CudaDevice>` handle; the device
//! must outlive all images created from it (C7). Raw-pointer extraction at the
//! FFI boundary follows the pattern documented in `docs/spike-cudarc-ptr-bridge.md`.

#![deny(missing_docs)]

/// Generated `impl ConvertTo` for all NPP-supported (src,dst) pairs (committed artifact).
pub mod convert_generated;
/// Macro to generate cross-type `impl ConvertTo` for image types.
pub mod convert_macros;
/// Placeholder module — both `ConvertTo` and `Normalize` are now generated
/// (`convert_generated.rs` and `normalize_generated.rs` respectively).
/// Retained for future hand-written conversion ops.
pub mod convert_ops;
/// Generated `impl ConvertRounded` for all narrowable (src,dst) pairs (committed artifact).
pub mod convert_round_generated;
/// Macro to generate `impl ConvertRounded` for cross-type rounding-mode conversions.
pub mod convert_round_macros;
/// Round-mode conversion helper — translates `RoundMode` to raw NPP constants.
pub mod convert_round_ops;
/// Generated `impl ConvertRoundedScaled` for all NPP-supported C1RSfs pairs (committed artifact).
pub mod convert_round_scaled_generated;
/// Macro to generate `impl ConvertRoundedScaled` for cross-type scaled rounding-mode conversions.
pub mod convert_round_scaled_macros;
pub mod cuda;
/// NPP error types and the `check_status` helper.
pub mod error;
/// Core GPU image type with `NppPixelType` marker trait.
pub mod image;
/// Capability traits (`Resize`, `SwapChannels`, `ConvertTo`, `ConvertRounded`, `ConvertRoundedScaled`, `Normalize`).
pub mod imageops;
/// Packed memory layout description.
pub mod layout;
/// Generated `impl Mean` for all supported types (committed artifact).
pub mod mean_generated;
/// Macro to generate `impl Mean` for image types.
pub mod mean_macros;
/// Generated `impl Normalize` for all NPP-supported integer→f32 pairs (committed artifact).
pub mod normalize_generated;
/// Macro to generate cross-type `impl Normalize` for image types.
pub mod normalize_macros;
/// GPU-probed (type, interpolation) support matrix for Resize.
pub mod resize_caps;
/// Generated `impl Resize` for all supported types (committed artifact).
pub mod resize_generated;
/// Macro to generate `impl Resize` for image types.
pub mod resize_macros;
/// Resize helper functions (`interpolation_mode`, `mode_supported`).
pub mod resize_ops;
pub mod stream;
pub use stream::{stream_context_for, StreamContext};
/// Generated `impl SwapChannels` for all supported types (committed artifact).
pub mod swap_channels_generated;
/// Macro to generate `impl SwapChannels` for image types.
pub mod swap_channels_macros;
/// Golden-test assertion helper (GPU-gated).
#[cfg(feature = "gpu")]
pub mod test_helpers;

#[cfg(test)]
mod raw_tests;

#[cfg(all(test, feature = "gpu"))]
mod resize_roi_tests;

#[cfg(all(test, feature = "gpu"))]
mod swap_channels_roi_tests;
