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
//! Cross-type operations ([`ConvertTo`](crate::imageops::ConvertTo), [`Normalize`](crate::imageops::Normalize))
//! convert between different pixel types (e.g. `u8 → f32`).
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

/// Macro to generate cross-type `impl ConvertTo` for image types.
pub mod convert_macros;
/// Generated `impl ConvertTo` for all NPP-supported (src,dst) pairs (committed artifact).
pub mod convert_generated;
/// Cross-type pixel format conversion operations (Normalize only).
///
/// ConvertTo is generated in `convert_generated.rs`.
/// Generalizing Normalize across the alphabet is deferred to F5.2.
pub mod convert_ops;
pub mod cuda;
/// NPP error types and the `check_status` helper.
pub mod error;
/// Core GPU image type with `NppPixelType` marker trait.
pub mod image;
/// Capability traits (`Resize`, `SwapChannels`, `ConvertTo`, `Normalize`).
pub mod imageops;
/// Packed memory layout description.
pub mod layout;
/// Generated `impl Mean` for all supported types (committed artifact).
pub mod mean_generated;
/// Macro to generate `impl Mean` for image types.
pub mod mean_macros;
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
