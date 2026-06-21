//! # npp-codegen — build-time tooling for `npp-rs`
//!
//! Reads NPP's bindgen output (`bindings.rs`), classifies symbols and
//! signatures, and emits Rust source for the runtime library. This crate
//! does **not** depend on `npp-rs` — the boundary is text+commit, not cargo.
//!
//! ## Crate roles
//! - **survey_shapes** (`bin`): produces the shape histogram.
//! - **classify**: symbol-name parser extracting (op, type, channel).
//! - **generators**: emit `impl_*_for!` invocation lists.
//!
//! See `docs/codegen-architecture.md` for the full architecture with mermaid
//! diagrams.

#![deny(missing_docs)]
