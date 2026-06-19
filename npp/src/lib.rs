//! Safe Rust bindings for NVIDIA NPP image operations.
//!
//! Built on:
//! - `cudarc` for CUDA device management (`Arc<CudaDevice>`, `CudaSlice<T>`)
//! - `npp-sys` for generated FFI bindings to the NPP image domain (`nppi*` symbols)
//!
//! The core type is [`CudaImage<T>`], where `T: NppPixelType` covers the full NPP
//! primitive alphabet (~9 types). Operation capability is expressed via traits
//! (e.g. [`Resize`], [`SwapChannels`]): an unsupported `(type, op)` pair simply
//! has no trait impl, making it a compile-time error.
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
//! The [`CudaImage`] constructor requires an `Arc<CudaDevice>` handle; the device
//! must outlive all images created from it (C7). Raw-pointer extraction at the
//! FFI boundary follows the pattern documented in `docs/spike-cudarc-ptr-bridge.md`.

#![deny(missing_docs)]

pub mod cuda;
/// NPP error types and the `check_status` helper.
pub mod error;
/// Core GPU image type with `NppPixelType` marker trait.
pub mod image;
/// Capability traits (`Resize`, `SwapChannels`).
pub mod imageops;
/// Packed memory layout description.
pub mod layout;
/// `Resize` implementations for `u8` and `f32`.
pub mod resize_ops;
/// `SwapChannels` implementation for `u8`.
pub mod swap_channel_ops;

#[cfg(test)]
mod raw_tests;
