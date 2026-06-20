//! GPU-probed (type, interpolation) support matrix for Resize.
//!
//! # Committed artifact — do NOT gitignore
//!
//! Unlike `bindings.rs` (gitignored, regenerated every build), this file
//! is **committed** because it requires a GPU to produce, and CI has no
//! GPU lane. Regenerate it with:
//!
//! ```bash
//! nix develop . --command cargo test --features gpu --test probe_resize_caps -- --nocapture
//! ```
//!
//! Then paste the printed literal below and copy the file.

use crate::imageops::ResizeInterpolation;

/// Supported (NPP type token, interpolation) pairs probed from NPP.
pub const RESIZE_CAPS: &[(&str, ResizeInterpolation)] = &[
    ("8u", ResizeInterpolation::NearestNeighbor),
    ("8u", ResizeInterpolation::Linear),
    ("8u", ResizeInterpolation::Cubic),
    ("8u", ResizeInterpolation::Super),
    ("8u", ResizeInterpolation::Lanczos),
    ("16u", ResizeInterpolation::NearestNeighbor),
    ("16u", ResizeInterpolation::Linear),
    ("16u", ResizeInterpolation::Cubic),
    ("16u", ResizeInterpolation::Super),
    ("16u", ResizeInterpolation::Lanczos),
    ("16s", ResizeInterpolation::NearestNeighbor),
    ("16s", ResizeInterpolation::Linear),
    ("16s", ResizeInterpolation::Cubic),
    ("16s", ResizeInterpolation::Super),
    ("16s", ResizeInterpolation::Lanczos),
    ("32f", ResizeInterpolation::NearestNeighbor),
    ("32f", ResizeInterpolation::Linear),
    ("32f", ResizeInterpolation::Cubic),
    ("32f", ResizeInterpolation::Super),
    ("32f", ResizeInterpolation::Lanczos),
];
