# Feature: F1 — macro-generated Resize binding codegen

## Architecture preamble

The plan below uses a specific design that differs from the brief on two points (resolved via interview):

| Item | Brief said | Plan uses |
|------|-----------|-----------|
| Codegen mechanism | "consumed at `macro_rules!` expansion time" (impossible — macros can't read files) | `macro_rules!` in source **+ a committed, generated invocation list** produced by `cargo run --example gen_resize_impls` |
| Mode safety (decision #7) | compile-time error for unsupported `(type, mode)` | **Runtime-checked** against committed `RESIZE_CAPS` table; existing `ResizeInterpolation` enum stays as-is |
| Probe output location | `data/resize_caps.rs` | `npp/src/resize_caps.rs` (normal module, `include!`'d) |

**Two structural facts the brief missed:**
1. **Channel count is runtime data, not a type** (`CudaImage::new(device, channels: u8, …)`, stored in `layout.channels`). So one `impl` per rust-type, with `match self.channels()` dispatching to `C1R/C3R/C4R` symbols.
2. **`16f` can't be driven from the safe layer** (`half` crate disabled, `NppPixelType` not impl'd for `f16`). The spike and probe use raw `npp_sys` FFI.

---

## Phase 1 — Capture the `nppiResize_*` symbol corpus

*Model runs the Nix build, extracts symbols, commits the fixture.* Committed corpus = shared input for the classifier's tests and the probe's symbol selection.

### Step 1.1 — Build to generate `bindings.rs`
Run `nix develop . --command cargo build`. This generates bindings into `OUT_DIR`. Locate the bindings file: `nix develop . --command bash -c 'find target -name bindings.rs -path "*/npp-sys/*"'`. Expect a path like `target/debug/build/npp-sys-<hash>/out/bindings.rs`.

### Step 1.2 — Extract Resize symbols
Extract every function declaration starting with `nppiResize`: `nix develop . --command bash -c 'rg -o "nppiResize[A-Za-z0-9_]*" <path> | sort -u > /tmp/resize_symbols.txt'`. Copy the file to `npp/tests/fixtures/nppiResize_symbols.txt`, adding a header line `# Captured from CUDA <version> via bindgen — regenerate on CUDA bump\n`.

### Step 1.3 — Commit and verify corpus
Create `npp/tests/fixtures/` if it doesn't exist. Verify the file is non-empty and contains at least `nppiResize_8u_C3R` and `nppiResize_32f_C3R`. Run `nix develop . --command cargo build` to confirm the crate still builds (the fixture itself isn't code, but the directory is now new).

**Verify:** `nix develop . --command cargo build -p npp-rs` passes. File `npp/tests/fixtures/nppiResize_symbols.txt` exists, non-empty, one symbol per line.

---

## Phase 2 — Suffix classifier module (pure Rust, no GPU)

*The core small-model sweet spot: a pure string-parsing `fn classify(symbols: &[&str]) -> Vec<ClassifiedSymbol>` with offline tests.*

### Step 2.1 — Create the output type and module

Create `npp/src/suffix_classifier.rs`:
```rust
/// A classified NPP symbol that fits the simple type×channel grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedSymbol {
    /// The raw FFI symbol, e.g. "nppiResize_8u_C3R".
    pub raw: String,
    /// Operation family, e.g. "Resize".
    pub op: String,
    /// NPP element-type token, e.g. "8u", "32f".
    pub type_token: String,
    /// Channel count (1, 3, or 4).
    pub channels: u8,
}

/// Classify NPP Resize symbols into the type×channel grid.
///
/// Only `nppiResize_<type>_C<n>R` (C1/C3/C4 only) is accepted.
/// SqrPixel, Batch, Advanced, AC4, P{n}, C2, Sfs, Ctx variants are rejected.
/// Rejected symbols are simply omitted from the returned `Vec`.
pub fn classify(symbols: &[&str]) -> Vec<ClassifiedSymbol> {
    symbols
        .iter()
        .filter_map(|raw| {
            let s = raw.strip_prefix("nppiResize_")?;
            // Reject any symbol with extra keywords after "Resize_"
            if !s.starts_with(|c: char| c.is_ascii_digit() || c == 'f') {
                return None; // e.g. "nppiResizeSqrPixel_8u_C1R"
            }
            let (type_token, rest) = s.split_once('_')?;
            // Validate type_token
            if !["8u","8s","16u","16s","32u","32s","32f","64f","16f"].contains(&type_token) {
                return None;
            }
            // Channel part must be C1, C3, or C4, ending with R
            let channels = match rest {
                "C1R" => 1,
                "C3R" => 3,
                "C4R" => 4,
                _ => return None, // rejects AC4R, P3R, C2R, C4C3R, ...C3RSfs, ...
            };
            Some(ClassifiedSymbol {
                raw: raw.to_string(),
                op: "Resize".to_string(),
                type_token: type_token.to_string(),
                channels,
            })
        })
        .collect()
}
```

### Step 2.2 — Declare the module in `lib.rs`
In `npp/src/lib.rs`, add a line like `/// Suffix classifier for NPP symbol names.` then `pub mod suffix_classifier;` (keep alphabetical order with `pub mod`s).

### Step 2.3 — Add offline unit tests in `suffix_classifier.rs`
Inside the same file, `#[cfg(test)] mod tests { use super::*; }` with:
- **Accept test:** `classify(&["nppiResize_8u_C3R"])` returns one `ClassifiedSymbol` with `type_token=="8u"`, `channels==3`, `raw=="nppiResize_8u_C3R"`. Repeat for `_16u_C1R`, `_32f_C4R`, `_8s_C3R`, `_64f_C1R`, `_16f_C3R` (1 each for breadth).
- **Reject test:** each of these returns empty `Vec`:
  - `nppiResizeSqrPixel_8u_C1R` (extra family word)
  - `nppiResizeBatch_8u_C1R` (extra family word)
  - `nppiResize_8u_AC4R` (AC4, not simple channel)
  - `nppiResize_8u_P3R` (planar)
  - `nppiResize_8u_C2R` (no C2 resize)
  - `nppiResize_8u_C4C3R` (channel-changing — different family)
  - `nppiResize_8u_C3RSfs` (extra suffix)
  - `nppiResize_32f` (no channel suffix)
  - `nppiResize_8u_C3Rtx` (extra suffix)
  - `nppiResize_zzz_C3R` (invalid type_token)
- **Fixture test:** `include_str!("../tests/fixtures/nppiResize_symbols.txt")`, split lines (skip `#` line and blanks), run classify, assert result is *non-empty*, assert every `channels ∈ {1,3,4}`, every `type_token` is known. Do NOT assert an exact count (CUDA-version-dependent).

### Step 2.4 — Build, test, clippy, fmt
Run the standard triad inside the Nix shell:
```
nix develop . --command cargo test -p npp-rs
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo fmt --check
```
All must pass. No GPU involved.

---

## Phase 3 — Status-code spike (GPU, capture-then-pin)

*Empirically establishes the closed NppStatus taxonomy for the probe. Reuses the `golden_resize.rs` "print output, human pins it" ritual. This test is kept as a committed CUDA-bump regression guard.*

### Step 3.1 — Add `PartialEq, Eq` to `ResizeInterpolation`
In `npp/src/imageops.rs`, change `#[derive(Debug, Clone, Copy)]` on `ResizeInterpolation` to `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`. Needed for both the cap-check and the probe.

### Step 3.2 — Create the spike test
Create `npp/tests/spike_npp_status.rs`, first line `#![cfg(feature = "gpu")]`. Use **raw `npp_sys` FFI** (not the safe `CudaImage` layer — allocate with `nppiMalloc_*`, free with `nppiFree`), mimicking `raw_tests.rs`. Call `nppiResize_*` three ways and `eprintln!` + `assert_eq!` the returned `NppStatus`:

- **Positive:** `nppiResize_32f_C3R` on 64×64→32×32, `NPPI_INTER_LANCZOS`. Assert `>= 0`.
- **Negative (mode unsupported):** pick a `(type, mode)` from the corpus that is documented-unsupported. Candidate: `nppiResize_16f_C3R` + Lanczos **iff** Phase 1 shows 16f symbols exist; otherwise pick another known unsupported pair. Assert the exact negative code (capture from first run).
- **Harness-bug (too-small image):** `nppiResize_32f_C3R` Lanczos on 1×1→1×1. Assert a step/size error code distinct from the negative code (capture from first run).

### Step 3.3 — Pin the taxonomy
First run: `nix develop . --command cargo test --features gpu --test spike_npp_status -- --nocapture`. It will fail because the pinned codes aren't set yet. Read the three printed codes from the output. **Hardcode the exact codes into the test as `const EXPECT_*`** (e.g. `const EXPECT_NEGATIVE: i32 = -1000;`).

Second run: the test should pass with the assertions against the pinned codes. This makes it a regression guard: if a CUDA bump changes the codes, the test fails and the model knows the probe's classification logic must be re-validated.

### Step 3.4 — Add a taxonomy doc-comment table at the top of `spike_npp_status.rs`
```rust
// Status-code taxonomy (pinned from CUDA <version> — update on CUDA bump):
//   NPP_SUCCESS (≥ 0) → supported
//   NPP_INTERPOLATION_ERROR (< 0) → mode unsupported
//   NPP_STEP_ERROR / NPP_SIZE_ERROR (< 0) → harness bug (image too small)
//   anything else → FAIL LOUD (spike was incomplete)
```

**Verify:** `nix develop . --command cargo test --features gpu --test spike_npp_status` passes on the GPU box. Plain `cargo test` skips it (gpu-gated).

---

## Phase 4 — Probe harness → `RESIZE_CAPS` (GPU, capture-then-pin)

*Produces the committed `(type, mode)` support table the runtime guard checks.*

### Step 4.1 — Create the probe test
Create `npp/tests/probe_resize_caps.rs`, first line `#![cfg(feature = "gpu")]`. Import `npp_sys` raw symbols, `ResizeInterpolation`.

Iterate over every `type_token` from the NPP alphabet that appears in the corpus AND that the safe layer supports (skip `16f`). For each, for each `ResizeInterpolation` variant (NN, Linear, Cubic, Super, Lanczos):

1. Allocate src/dst with `nppiMalloc_<t>_C3` (representative channel, use the type token from the symbol name to infer the NPP allocation fn — e.g. `8u` → `nppiMalloc_8u_C3`). Free with `nppiFree`.
2. Call the corresponding `nppiResize_<t>_C3R` on a **comfortably-sized** image (e.g. 64×64 → 32×32 — large enough to avoid the step/size error).
3. Classify the result using the Phase 3 taxonomy:
   - `>= 0` → record as supported
   - `== NEGATIVE_CODE` → record as unsupported
   - `== STEP_SIZE_CODE` → **panic** "probe image too small"
   - anything else → **panic** "unknown status — spike was incomplete"
4. Accumulate supported pairs.

After all iterations, `eprintln!("{:?}", supported_pairs)` formatted as a ready-to-paste Rust array literal: `[("8u", ResizeInterpolation::Linear), ...]`.

### Step 4.2 — Pin the table into `resize_caps.rs`
First run: `nix develop . --command cargo test --features gpu --test probe_resize_caps -- --nocapture`. Read the printed array literal.

Create `npp/src/resize_caps.rs`:
```rust
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
    // ── paste the literal from the probe test here ──
];
```

Declare the module in `lib.rs`: `pub mod resize_caps;` (with doc comment).

### Step 4.3 — Verify the module compiles
`nix develop . --command cargo build -p npp-rs` must pass. The table may be empty or partial at first — that's fine, it compiles either way. The probe test on GPU must pass.

**Verify:** `cargo build -p npp-rs` passes (no GPU). `cargo test --features gpu --test probe_resize_caps` passes on GPU.

---

## Phase 5 — Macro + generated invocations + runtime cap-check (pure Rust)

*Replaces the two hand-written `Resize` impls with macro-generated ones covering the whole corpus, with the runtime mode guard.*

### Step 5.1 — Add the shared `mode_supported` helper
In `npp/src/resize_ops.rs`, add:
```rust
/// Returns true iff the probe confirmed `(type_token, inter)` is supported.
fn mode_supported(type_token: &str, inter: ResizeInterpolation) -> bool {
    crate::resize_caps::RESIZE_CAPS
        .iter()
        .any(|(t, m)| *t == type_token && *m == inter)
}
```
Verify this compiles: `cargo build -p npp-rs`.

### Step 5.2 — Write `impl_resize_for!` macro
Create `npp/src/resize_macros.rs`:
```rust
/// Macro: generate `impl Resize for CudaImage<$rust_ty>`.
///
/// # Arguments
///
/// * `$rust_ty` — the Rust pixel element type (e.g. `u8`, `f32`).
/// * `$token` — the NPP type token string (e.g. `"8u"`, `"32f"`).
/// * `{$($ch:literal => $sym:path),+}` — channel-count arms mapping to NPP symbols,
///   e.g. `{ 3 => npp_sys::nppiResize_8u_C3R, 4 => npp_sys::nppiResize_8u_C4R }`.
macro_rules! impl_resize_for {
    ($rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
        // ... body ...
    };
}
```
The macro expands to `impl Resize for CudaImage<$rust_ty>` with `fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError>`:
1. Runtime guard: `if !mode_supported($token, inter) { return Err(NppError::InvalidArgument(format!(…))) }`
2. Build `NppiSize` / `NppiRect` from `self.width()/height()` and `dst.width()/height()` — exactly as the current `resize_ops.rs` body.
3. Byte-step: `(self.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32` (and same for `dst`).
4. Pointer bridge: `cudarc::driver::DevicePtr::device_ptr(&self.buf)` + `img_index` offset — paste the exact pointer arithmetic from the f32 impl (`resize_ops.rs:145-148`).
5. `match self.channels() { $($ch => unsafe { $sym(…, interpolation_mode(inter)) }),+  … => Err(…) }`.
6. `check_status(status)`.
7. Front-load the doc comment: explain nStep-in-bytes, non-overlap precondition, sub-image correctness via img_index.

Import everything the body needs in the macro's expansion (`crate::error::{check_status, NppError}`, `crate::image::CudaImage`, `crate::imageops::{Resize, ResizeInterpolation}`, `crate::resize_caps`, `npp_sys::{…}`, etc.).

Declare the module in `lib.rs`: `pub mod resize_macros;` (with doc comment). `pub use impl_resize_for;` if needed or just scope it (the generated file will be in the same crate, so `macro_rules!` is visible after the module's `#[macro_use]` or by direct reference — the simplest is to use `#[macro_export]` so the macro is accessible crate-wide, or just declare it at the crate root. For clarity, add `#[macro_export]` to the `macro_rules!` definition so it's in the crate's root namespace.

### Step 5.3 — Generate `resize_generated.rs`
Create `npp/examples/gen_resize_impls.rs` (no additional config needed — `Cargo.toml` already has `autobenches = false` but `[lib]` name is default, examples work). The example:
1. Include the fixture: `include_str!("../tests/fixtures/nppiResize_symbols.txt")`.
2. Split lines, skip comments/blanks, call `npp_rs::suffix_classifier::classify`.
3. **Group** results by `type_token`.
4. Map `type_token` to `rust_ty` via a match: `"8u" => (u8), "8s" => (i8), "16u" => (u16), "16s" => (i16), "32u" => (u32), "32s" => (i32), "32f" => (f32), "64f" => (f64)`. Skip `16f` (while `half` is disabled).
5. For each group, emit one `impl_resize_for!(<rust_ty>, "<token>", { <ch> => npp_sys::nppiResize_<token>_C<ch>R, … });` — space-delimited, one arm per channel.
6. Print to stdout.

Run: `nix develop . --command cargo run -p npp-rs --example gen_resize_impls > /tmp/resize_generated.rs`. Inspect the output — it should produce one `impl_resize_for!` per type, e.g.:
```rust
impl_resize_for!(u8, "8u", { 1 => npp_sys::nppiResize_8u_C1R, 3 => npp_sys::nppiResize_8u_C3R, 4 => npp_sys::nppiResize_8u_C4R });
```

### Step 5.4 — Create and consume `resize_generated.rs`
Copy the output to `npp/src/resize_generated.rs`, prefixed with:
```rust
//! GENERATED — re-run `cargo run --example gen_resize_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::{error::NppError, image::CudaImage, imageops::{Resize, ResizeInterpolation}};

// ── paste the invocation list here ──
```

Declare the module in `lib.rs`: `pub mod resize_generated;` (with doc comment).

### Step 5.5 — Delete the hand-written impls
In `npp/src/resize_ops.rs`, delete the two `impl Resize for CudaImage<u8>` and `impl Resize for CudaImage<f32>` blocks (keep the `interpolation_mode` helper function, the new `mode_supported` helper, and the doc-comment at the top explaining nStep-in-bytes). Remove the individual imports (`nppiResize_8u_C3R`, `nppiResize_32f_C3R`) from the `use` statement — `npp_sys` will be imported by the generated file or a general `use npp_sys::*`. Adjust to keep only what `interpolation_mode` and `mode_supported` need.

### Step 5.6 — Non-GPU regression test for the guard
In `resize_ops.rs` (or a new test module), add a non-GPU `#[test]`:
```rust
#[test]
fn test_mode_unsupported_returns_false() {
    // Pick a (type, mode) the probe certainly won't list — e.g. "8u" + Super
    // (Super is rarely supported for integer types). This tests the guard logic
    // path without needing a GPU.
    assert_eq!(mode_supported("8u", ResizeInterpolation::Super), false);
}
```
(If `RESIZE_CAPS` is empty/unpopulated from Phase 4.2, this test may trivially pass. When the table is populated, it becomes a meaningful regression guard.)

### Step 5.7 — Full verification
```
nix develop . --command cargo build -p npp-rs          # All generated impls compile against real npp_sys symbols
nix develop . --command cargo test -p npp-rs            # Classifier + guard tests, no GPU
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo fmt --check
nix develop . --command cargo run --example gen_resize_impls   # Re-runnable, matches committed output
nix develop . --command cargo test --features gpu -p npp-rs    # golden_resize passes through generated u8 impl
```

---

## Phase 6 — Docs reconciliation

### Step 6.1 — Update `.spar/brief.md`
- Mark decision #7 as superseded (runtime-checked mode, see plan).
- Add the channel-is-runtime-dispatch structural fact.
- Add `npp/src/resize_generated.rs`, `npp/src/resize_caps.rs` to artifact table (instead of `data/resize_caps.rs`).
- Note the spike is kept as CUDA-bump regression guard (remove "throwaway" framing).
- Note the status-code spike's negative case is corpus-selected, not hardcoded to 16f.

### Step 6.2 — Update `docs/roadmap.md` F1 section
Replace the current F1 description with the final architecture: the two-pipeline design (scrape `bindings.rs` → classifier → generated invocations → macro; GPU probe → committed caps table → runtime guard), the committed/gitignored artifact policy per artifact, the spike dependency, and the plan's resolution of the compile-time-vs-runtime mode question.

### Step 6.3 — Update `docs/architecture.md` module table
Add `suffix_classifier`, `resize_macros`, `resize_caps`, `resize_generated` to the module table. Note `resize_caps` and `resize_generated` are committed generated artifacts.

**Verify:** `nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys` builds, `nix develop . --command cargo fmt --check` passes.
