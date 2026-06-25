# Feature: F5.2 — Normalize codegen across the integer→f32 alphabet

## Goal (restated)

Replace the hand-written `Normalize<f32> for CudaImage<u8>` with **generated
impls covering every integer-source type for which a `ConvertTo<f32>` impl
already exists** — currently `u8`, `u16`, `i16`. No public API change.

> The deliverable set is **determined by the F5.1 `nppiConvert_symbols.txt`
> fixture**, not by this plan: a source type receives a `Normalize<f32>` impl iff
> its `(src, "32f")` Convert symbol pair exists. "Generalize across the alphabet"
> therefore means "every such pair," not any specific type guarantee.

## Key architectural decision (differs from F5.1 pattern)

Unlike the four existing generated families (Resize, SwapChannels, Mean,
Convert), Normalize does **not** use a `FamilyDescriptor` + `generate_for_family()`.
It uses its own standalone generator (`generate_normalize_impls`) that filters the
Convert fixture to `dst == "32f"` pairs and emits trivial three-argument macro
invocations. The convert step delegates to the `ConvertTo` trait; the three MulC
symbols are **hardcoded in the macro body** (always `nppiMulC_32f_C1/3/4R_Ctx`).
This avoids the fragile `dual_type` hijacking and the `[c; $ch]` compilation bug
that would result from the F5.1 pattern.

## Scale-constant model (Option B, resolved)

| Source | Token | Denominator | Rationale |
|--------|-------|-------------|-----------|
| `u8` | `"8u"` | `255.0_f32` | `2^8 - 1` |
| `u16` | `"16u"` | `65535.0_f32` | `2^16 - 1` |
| `i16` | `"16s"` | `32767.0_f32` | `2^15 - 1` (max positive) |
| `i8` | `"8s"` | `127.0_f32` | `2^7 - 1` (max positive) |
| `u32`/`i32` | — | (defined in helper, not emitted yet) | Latent for future fixture growth |
| `32f`/`64f` | — | `None` | Float sources excluded — no defined denominator |

**Contract:** The maximum positive representable value maps to exactly `1.0`.
Negative inputs (signed types) map below `0.0` (e.g. `-32768 → ~-1.000031`).

---

## Phase 1: The `impl_normalize_for!` macro

Commit message: `feat(npp-rs): add impl_normalize_for macro for cross-type normalization`

### Step 1: Write the macro

Create `npp/src/normalize_macros.rs` defining:

```rust
#[macro_export]
macro_rules! impl_normalize_for {
    ($src_ty:ty, $denominator:expr, $src_token:expr) => { … };
}
```

**Signature — three arguments only**, no channel arms, no symbol paths:
- `$src_ty` — the Rust source element type (e.g. `u8`, `u16`, `i16`).
- `$denominator` — the f32 literal for the max positive value (e.g. `255.0_f32`).
- `$src_token` — the NPP source type token string (e.g. `"8u"`), used for
  error messages and doc generation.

**Expansion** — `impl Normalize<f32> for CudaImage<$src_ty>` with
`fn normalize(&self, dst: &mut CudaImage<f32>) -> Result<(), NppError>`:

1. **Convert step** — delegate to the `ConvertTo` trait:

   ```rust
   self.convert(dst)?;
   ```

   This resolves to the generated `ConvertTo<f32> for CudaImage<$src_ty>` in
   `convert_generated.rs`, which handles dimension validation, pointer extraction,
   step calculation, sub-image `img_index` offset, and channel dispatch. The trait
   method does its own dimension/channel matching — there is **no redundant
   leading validation** in this macro (matches the hand-written reference at
   `convert_ops.rs:42–44`).

2. **Scale step** — in-place `MulC` on `dst` with constant
   `1.0_f32 / $denominator`:

   ```rust
   let scale: f32 = 1.0_f32 / $denominator;

   let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
   let dst_ptr = (*dst_base + dst.layout.img_index as u64) as *mut f32;

   let dst_step = (dst.layout.height_stride * std::mem::size_of::<f32>()) as i32;

   let size = npp_sys::NppiSize {
       width: dst.width() as i32,
       height: dst.height() as i32,
   };
   ```

   Then explicit per-arm dispatch (NOT `[c; $ch]` — that would fail compilation
   because a `macro_rules!` literal cannot serve as an array repeat count):

   ```rust
   let status = unsafe {
       match dst.channels() {
           1 => {
               npp_sys::nppiMulC_32f_C1R_Ctx(
                   dst_ptr, dst_step, scale,
                   dst_ptr, dst_step, size, dst.ctx.raw_ctx(),
               )
           }
           3 => {
               let constants: [f32; 3] = [scale; 3];
               npp_sys::nppiMulC_32f_C3R_Ctx(
                   dst_ptr, dst_step, constants.as_ptr(),
                   dst_ptr, dst_step, size, dst.ctx.raw_ctx(),
               )
           }
           4 => {
               let constants: [f32; 4] = [scale; 4];
               npp_sys::nppiMulC_32f_C4R_Ctx(
                   dst_ptr, dst_step, constants.as_ptr(),
                   dst_ptr, dst_step, size, dst.ctx.raw_ctx(),
               )
           }
           _ => {
               return Err(NppError::InvalidArgument(format!(
                   "unsupported channel count {} for normalize {}→f32",
                   dst.channels(), $src_token,
               )));
           }
       }
   };
   check_status(status)?;
   Ok(())
   ```

   **Defensive Err over unreachable:** the `_ => Err(...)` form matches the
   generated-macro convention (`convert_macros.rs:119`) rather than the
   hand-written code's `unreachable!()` (`convert_ops.rs:93`). Since
   `self.convert(dst)?` runs first and validates that `dst.channels()` is in
   {1,3,4} (the convert impl's own arm set), the Err arm is genuinely unreachable
   in normal operation — but the Err form is more defensive in unsafe-adjacent
   macro-generated code.

**In-place aliasing note:** `dst_ptr` is passed as both src and dst to `MulC`.
This is sound because `MulC` is purely elementwise — output pixel (x,y) depends
only on input pixel (x,y). No gather-overlap hazard (C4). This is exactly how the
hand-written `Normalize` works at `convert_ops.rs:70–93`.

**Annotations and docs:**
- `#[allow(clippy::macro_metavars_in_unsafe)]` on the `impl` block (matching
  `convert_macros.rs:75`).
- Do **NOT** add `#[cfg(not(tarpaulin_include))]` — match the four existing
  generated macros which carry no such annotation (finding W6, ratified in F5.1).
- `#[macro_export]`-level doc comment documenting: arguments; the two-leg
  convert-then-scale expansion; that `$denominator` is the source type's maximum
  positive representable value; that the convert leg delegates to `ConvertTo<f32>`
  (non-overlap precondition inherited); that the MulC leg is purely elementwise
  and safely aliases; sub-image support via `img_index` offset; the C7
  device-lifetime invariant; and: "Normalization of float source types is
  excluded (denominator undefined). For signed types, negative inputs map below
  `0.0`."

### Step 2: Wire into `lib.rs`

In `npp/src/lib.rs`, add (in **alphabetical** position among module declarations;
note this puts it after `mean_*` and before `resize_caps`):

```rust
/// Macro to generate cross-type `impl Normalize` for image types.
pub mod normalize_macros;
/// Generated `impl Normalize` for all NPP-supported integer→f32 pairs (committed artifact).
pub mod normalize_generated;
```

Update the `pub mod convert_ops;` module doc at line 36–39 from:
```rust
/// Cross-type pixel format conversion operations (Normalize only).
///
/// ConvertTo is generated in `convert_generated.rs`.
/// Generalizing Normalize across the alphabet is deferred to F5.2.
```
to:
```rust
/// Placeholder module — both `ConvertTo` and `Normalize` are now generated
/// (`convert_generated.rs` and `normalize_generated.rs` respectively).
/// Retained for future hand-written conversion ops.
/// Rounding-mode ConvertTo variants are deferred to **F5.3**.
```

Update the crate-level doc at `lib.rs:12` from:
```
//! Cross-type operations ([`ConvertTo`](...), [`Normalize`](...))
//! convert between different pixel types (e.g. `u8 → f32`).
```
to append a sentence: "Normalize is generated for integer→f32 pairs; rounding-mode
ConvertTo is deferred to F5.3."

### Step 3: Verify the expanded crate compiles with use statements

At this point `normalize_generated.rs` doesn't exist yet (it lives in Phase 3).
The `pub mod normalize_generated;` in `lib.rs` will fail to compile. That's
expected — the module declaration lands in Phase 1, the file lands in Phase 3.
Alternatively, if you prefer every commit green, land the module declaration in
Phase 3 alongside the file. The latter is cleaner. Either way: **do not commit
a non-compiling git state.**

**Recommended:** skip the `pub mod` lines in this commit. Land them in Phase 3
Step 1 (the placeholder commit) alongside the first `normalize_generated.rs`.
The commit message already includes both steps.

---

## Phase 2: Normalize generator in npp-codegen

Commit message: `feat(npp-codegen): add normalize generator and scale-denominator helper`

### Step 1: Add `normalize_scale_denominator` to `gen_impls.rs`

In `npp-codegen/src/gen_impls.rs`, add:

```rust
/// Return the f32 normalization denominator for an NPP integer type token.
///
/// The denominator is the maximum positive representable value for the type,
/// such that calling `MulC` with `1/denominator` maps the type's maximum
/// positive value to exactly `1.0`:
///
/// | Token | Type   | BITS | Formula         | Value            |
/// |-------|--------|------|-----------------|------------------|
/// | 8u    | `u8`   | 8    | `2^8 - 1`       | 255              |
/// | 8s    | `i8`   | 8    | `2^(8-1) - 1`   | 127              |
/// | 16u   | `u16`  | 16   | `2^16 - 1`      | 65535            |
/// | 16s   | `i16`  | 16   | `2^(16-1) - 1`  | 32767            |
/// | 32u   | `u32`  | 32   | `2^32 - 1`      | 4294967295       |
/// | 32s   | `i32`  | 32   | `2^(32-1) - 1`  | 2147483647       |
/// | 32f   | `f32`  | —    | —               | `None` (excluded)|
/// | 64f   | `f64`  | —    | —               | `None` (excluded)|
///
/// Float sources are excluded because there is no canonical denominator —
/// they are already in floating-point range. These tokens are returned as
/// `f64` to avoid precision loss in the internal lookup; the generator
/// formats them as `{val}_f32` string literals for the emitted source.
///
/// ## Precision note (forward-looking)
///
/// The u32 and i32 denominators (`4294967295`, `2147483647`) are **not exactly
/// representable as f32**, and emitting them as `4294967295.0_f32` would trigger
/// `clippy::excessive_precision`. They are included in this helper for
/// completeness but are **not emitted today** — only u8/u16/i16 have Convert→f32
/// symbols. If a future CUDA bump adds u32→f32 or i32→f32 Convert symbols, the
/// generator must emit those denominators as a `const` cast (`u32::MAX as f32`)
/// rather than a decimal float literal.
pub fn normalize_scale_denominator(token: &str) -> Option<f64> {
    match token {
        "8u" => Some(255.0),
        "8s" => Some(127.0),
        "16u" => Some(65535.0),
        "16s" => Some(32767.0),
        "32u" => Some(4_294_967_295.0),
        "32s" => Some(2_147_483_647.0),
        "32f" | "64f" => None,
        _ => None,
    }
}
```

Add a `#[cfg(test)]` unit test `normalize_denominators`:
- Assert `"8u"` → `Some(255.0)`
- Assert `"16u"` → `Some(65535.0)`
- Assert `"16s"` → `Some(32767.0)`
- Assert `"8s"` → `Some(127.0)`
- Assert `"32f"` → `None`
- Assert `"64f"` → `None`
- Assert unknown `"zzz"` → `None`

### Step 2: Add `generate_normalize_impls` function

In `npp-codegen/src/gen_impls.rs`, add a **standalone** function (does NOT use
`FamilyDescriptor` or `generate_for_family`):

```rust
/// Generate `impl_normalize_for!` invocations for all integer→f32 pairs.
///
/// Reads the Convert fixture (same symbols as `CONVERT_FAMILY`), classifies
/// them, filters to pairs where `dst_token == "32f"` and the source has a
/// defined `normalize_scale_denominator`, and emits one invocation per
/// source type. The macro signature is trivially self-contained — no channel
/// arms or symbol tuples needed because the convert step delegates to the
/// `ConvertTo` trait and the MulC symbols are hardcoded in the macro body.
///
/// Unlike the other four families, this does NOT use `generate_for_family()`.
/// The Normalize generator has a completely different emit shape.
pub fn generate_normalize_impls(symbols: &[String]) -> String { ... }
```

Implementation details:

1. Convert `symbols` to `&[&str]` refs.
2. Call `classify_convert(&symbol_refs)`.
3. Build a `BTreeMap<&str, Option<f64>>` from `src_token` to its
   `normalize_scale_denominator`, keyed only for entries where
   `cs.dst_type_token == Some("32f")` and `skip_16f` is satisfied
   (reject `src_token == "16f"`). Use `BTreeMap` for deterministic ordering.
   Multiple routing entries for the same `src_token` (C1, C3, C4) collapse to
   one key.
4. Skip any token where `npp_type_to_rust(token).is_none()` or
   `normalize_scale_denominator(token).is_none()`.
5. Emit the standard generated header:
   ```rust
   //! GENERATED — re-run `cargo run --example gen_normalize_impls` on CUDA bump.
   //! This file is **committed** (like `resize_caps.rs`), not gitignored.
   ```
6. Emit use statements (these must match what the macro body needs — note
   `ConvertTo` and `DevicePtrMut` are required because the macro body calls
   `self.convert(dst)?` and extracts `DevicePtrMut`):
   ```rust
   use crate::error::{check_status, NppError};
   use crate::image::CudaImage;
   use crate::imageops::{ConvertTo, Normalize};
   use crate::impl_normalize_for;
   use cudarc::driver::DevicePtrMut;
   use npp_sys::NppiSize;
   use std::mem::size_of;
   ```
7. For each surviving `(token, Some(denom))` in sort order, emit:
   ```rust
   impl_normalize_for!(u8, 255.0_f32, "8u");
   ```
   Denomination formatting: `format!("{denom}_f32")` where `denom` is an `f64`.
   For u8/u16/i16, this produces `255.0_f32` / `65535.0_f32` / `32767.0_f32`.
   All three are exactly f32-representable.

### Step 3: Add generator unit tests

In `npp-codegen/src/gen_impls.rs` `#[cfg(test)]`, add
`generate_normalize_from_fixture`:

```rust
#[test]
fn generate_normalize_from_fixture() {
    let fixture = fixture_path("nppiConvert_symbols.txt");
    let (symbols, _) = read_fixture(&fixture);
    assert!(!symbols.is_empty(), "Convert fixture must not be empty");
    let generated = generate_normalize_impls(&symbols);
    assert!(!generated.is_empty(), "generated output must not be empty");

    // Must contain u8→f32 (8u has denominator 255)
    assert!(generated.contains("impl_normalize_for!(u8, 255.0_f32, \"8u\");"));
    // Must contain u16→f32
    assert!(generated.contains("impl_normalize_for!(u16, 65535.0_f32, \"16u\");"));
    // Must contain i16→f32
    assert!(generated.contains("impl_normalize_for!(i16, 32767.0_f32, \"16s\");"));
    // Must NOT contain non-f32 destinations (e.g. u16→i32)
    assert!(!generated.contains("impl_normalize_for!(u16,"), "u16 should only appear with 32f dst");
    // Must NOT contain 16f / f16
    assert!(!generated.contains("16f"));
    assert!(!generated.contains("f16"));
    // Must NOT contain float source types
    assert!(!generated.contains("impl_normalize_for!(f32"));
    assert!(!generated.contains("impl_normalize_for!(f64"));
    // Must include a ConvertTo import (macro body calls self.convert() via trait)
    assert!(generated.contains("use crate::imageops::ConvertTo;"));
    // Must include DevicePtrMut (macro body extracts dst pointer)
    assert!(generated.contains("use cudarc::driver::DevicePtrMut;"));
}
```

### Step 4: Add the `gen_normalize_impls` generator example

Create `npp-codegen/examples/gen_normalize_impls.rs`:

```rust
//! Generate `npp/src/normalize_generated.rs` from the Convert fixture.
//!
//! Reuses `nppiConvert_symbols.txt` (the same fixture as Convert) because
//! Normalize is a derived operation — it only wraps integer→f32 Convert pairs
//! with a scale step.
//!
//! Run: `cargo run --example gen_normalize_impls`

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir
        .join("tests")
        .join("fixtures")
        .join("nppiConvert_symbols.txt");
    let (symbols, _) = npp_codegen::gen_impls::read_fixture(&fixture);

    let generated = npp_codegen::gen_impls::generate_normalize_impls(&symbols);

    let out_path = manifest_dir
        .parent()
        .unwrap()
        .join("npp")
        .join("src")
        .join("normalize_generated.rs");

    std::fs::write(&out_path, generated.as_bytes())
        .expect("failed to write normalize_generated.rs");

    eprintln!("Wrote {}", out_path.display());
}
```

---

## Phase 3: Committed placeholder and hand-written impl removal

Commit message: `feat(npp-rs): wire normalize_generated, remove hand-written Normalize`

### Step 1: Create `normalize_generated.rs` placeholder

Create `npp/src/normalize_generated.rs` with:

```rust
//! GENERATED — re-run `cargo run --example gen_normalize_impls` on CUDA bump.
//! This file is **committed** (like `resize_caps.rs`), not gitignored.

use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{ConvertTo, Normalize};
use crate::impl_normalize_for;
use cudarc::driver::DevicePtrMut;
use npp_sys::NppiSize;
use std::mem::size_of;

impl_normalize_for!(u8, 255.0_f32, "8u");
```

This is the minimal file that makes the crate compile with the
`pub mod normalize_generated;` declaration from Phase 1 Step 2 and the existing
`golden_normalize.rs` test (which only exercises u8→f32).

If you deferred the `pub mod` lines to this commit (recommended), add them here
in `lib.rs` along with the file. **Verify** that `cargo build -p npp-rs` compiles
and `cargo test -p npp-rs --lib` passes.

### Step 2: Remove the hand-written `Normalize` from `convert_ops.rs`

In `npp/src/convert_ops.rs`:

- **Remove** the entire `impl Normalize<f32> for CudaImage<u8>` block
  (lines 40–99 of the current file).
- **Remove** now-unused imports: `{ConvertTo, Normalize}`, `check_status`,
  `NppError`, `CudaImage`, `DevicePtrMut`, `NppiSize`, `size_of`. After removal,
  the file may be left with only the module doc — keep `pub mod convert_ops;`
  wiring as a doc anchor.
- **Update the module doc** to:

  ```rust
  //! Placeholder module — both `ConvertTo` and `Normalize` are now generated
  //! (`convert_generated.rs` and `normalize_generated.rs` respectively).
  //! Retained for future hand-written conversion ops.
  //! Rounding-mode ConvertTo variants are deferred to **F5.3**.
  ```

### Step 3: Call-site audit — confirm no other consumers

The grepped `Normalize` / `.normalize(` references are:
- `imageops.rs` trait definition — unchanged.
- `lib.rs` doc lines — updated in Phase 1 Step 2.
- `golden_normalize.rs` — imports the trait, calls `src.normalize(&mut dst)`.
  After removal, this resolves to the generated `impl Normalize<f32> for
  CudaImage<u8>` in `normalize_generated.rs`. No changes needed to the test.
- Doc references in other macros (`resize_macros.rs`, `swap_channels_macros.rs`,
  `convert_macros.rs`) use `Normalize` in prose ("see Normalize") — no code
  binding, no changes needed.

**Verify:** `cargo build -p npp-rs` compiles; `cargo test -p npp-rs --lib` passes.

---

## Phase 4: Regenerate, byte-identity guard, and verification

Commit message: `test(npp-rs): regenerate normalize_generated and add byte-identity guard`

### Step 1: Regenerate and build (MulC C4 symbol gate)

```bash
nix develop . --command cargo run -p npp-codegen --example gen_normalize_impls
nix develop . --command cargo build -p npp-rs
```

**This build is the `nppiMulC_32f_C4R_Ctx` symbol verification gate.**
If this symbol does not exist in the NPP bindings on this CUDA host, the build
fails with an unresolved path error. **Resolution if C4 fails:** remove the
`4 =>` arm from the macro body (restrict Normalize to C1/C3 only, matching the
hand-written code's original scope). Document the restriction in the macro doc.

### Step 2: Add the byte-identity guard test

In `npp-codegen/src/gen_impls.rs` `#[cfg(test)]`, add
`normalize_generated_is_byte_identical` mirroring
`convert_generated_is_byte_identical` (which itself is at lines 438–475):

```rust
#[test]
fn normalize_generated_is_byte_identical() {
    // Read the committed generated file
    let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("npp")
        .join("src")
        .join("normalize_generated.rs");
    let committed = fs::read_to_string(&committed_path)
        .expect("failed to read committed normalize_generated.rs");

    // Generate from the fixture
    let fixture = fixture_path("nppiConvert_symbols.txt");
    let (symbols, _) = read_fixture(&fixture);
    let generated = generate_normalize_impls(&symbols);

    // Show diff if not identical
    if committed != generated {
        // … same line-diff diagnostic as the existing tests …
    }

    assert_eq!(
        committed, generated,
        "normalize_generated.rs must be byte-identical"
    );
}
```

This test lands in the **same commit** as the regenerated file, so no commit is
ever red (matching F5.1 convention, `plan-f5.1.md:118`).

### Step 3: Verification checklist

Run all gates in the **Nix dev shell** (`nix develop . --command` for each):

| Gate | Command | Expected |
|------|---------|----------|
| Build | `cargo build` | Green (includes npp-sys bindgen) |
| Build npp-rs | `cargo build -p npp-rs` | Green (MulC symbol gate) |
| Non-GPU test | `cargo test -p npp-rs --lib` | Green |
| Codegen test | `cargo test -p npp-codegen` | Green (byte-identity + generator) |
| Lint | `cargo clippy -- -D warnings` | Clean |
| Format | `cargo fmt --check` | Clean |
| Coverage | `cargo tarpaulin` | ≥ 90% |
| Profanity | `convert_generated_is_byte_identical` | Stays green (Convert path untouched) |

**Coverage note:** The generated Normalize impls are FFI-dead under non-GPU
tarpaulin, exactly like the four existing families. If coverage drops, investigate
the cause **before** adding `#[cfg(not(tarpaulin_include))]` (the existing macros
don't use one; finding W6).

---

## Phase 5: Golden tests for generalized Normalize

Commit message: `test(npp-rs): add golden tests for 16u and 16s normalization`

### Step 1: Golden — `u16 → f32` normalize

Create `npp/tests/golden_normalize_16u32f.rs` (following the exact structure of
`golden_normalize.rs`):

```rust
//! Golden-image correctness test for `Normalize` on `CudaImage<u16>` → `CudaImage<f32>`.
//!
//! # To update the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_normalize_16u32f`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.
//!
//! # Numerical note
//!
//! The normalized output is `(x as f32) * (1.0 / 65535.0)` (multiply-by-reciprocal,
//! not `(x as f32) / 65535.0`). These can differ in the last ULP. Always pin
//! `EXPECTED` from an actual GPU run — do not fabricate by hand.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::Normalize;
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 12;
const H: u32 = 8;

fn make_input() -> Vec<u16> {
    let mut data = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            // Values 0, 32768, 65535 produce exact normalized outputs 0.0, ~0.5, 1.0
            let val: u16 = match (x + y) % 3 {
                0 => 0,
                1 => 32768,
                _ => 65535,
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Golden output for u16→f32 normalization (C3).
/// Pinned on <GPU model> on <date>.
const EXPECTED: &[f32] = &[]; // Pin on GPU host

#[test]
fn test_golden_normalize_16u32f_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let src = CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");
    let mut dst = CudaImage::<f32>::new(ctx.clone(), 3, W, H).expect("dst allocation");
    src.normalize(&mut dst).expect("normalize");
    let output: Vec<f32> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "normalize_16u32f_c3");
}
```

Scale-constant verification: `32768 / 65535 = 0.500007629…` in f32
(multiply-by-reciprocal). Must be pinned, not computed.

### Step 2: Golden — `i16 → f32` normalize (signed source)

Create `npp/tests/golden_normalize_16s32f.rs`, same structure with:

- Input type `i16`, denominator `32767.0_f32`
- Values `0`, `16384`, `32767` (non-negative only for pinning simplicity)
- Doc note: "Normalize maps the maximum positive value (32767) to exactly `1.0`.
  Negative inputs map below `0.0` (e.g. `-32768 → ~-1.000031`). This test uses
  non-negative inputs for pinning simplicity."
- `CSS expected placeholder; pin on GPU host

### Step 3: Verify existing tests still pass

- `golden_normalize.rs` — unchanged, still tests `u8→f32` via the generated impl.
- `golden_convert.rs` + `golden_convert_16u32f.rs` + `golden_convert_8u16u.rs` —
  untouched, pass unchanged.

---

## Phase 6: Documentation reconciliation and F5.3 split-out

Commit message: `docs: reconcile F5.2 Normalize codegen and add F5.3 entry`

### Step 1: Update `docs/codegen-architecture.md`

In the "Families implemented" table (currently lines 110–115), add a Normalize
row:

| Normalize | `nppiConvert_` (reused) | C1, C3, C4 | — | — | `SRC+STEP, DST+STEP, SIZE` | Yes (dual-type, reused from Convert) |

Add a "Normalize: convert-then-scale" subsection (after the "Dual-type families"
section) explaining:

- Unlike the other four families, Normalize does **not** use `FamilyDescriptor`
  or `generate_for_family()`. It has its own standalone generator
  (`generate_normalize_impls`) in `gen_impls.rs`.
- The generator filters the **existing** Convert fixture (`nppiConvert_symbols.txt`)
  to pairs where `dst_token == "32f"` and the source has a defined integer
  denominator. No separate fixture needed.
- The `impl_normalize_for!` macro has a **trivial signature**:
  `($src_ty, $denominator, $src_token)` — no channel arms or symbol tuples.
  The convert step calls `self.convert(dst)?` via the `ConvertTo` trait; the
  MulC scale step is hardcoded inside the macro body using `nppiMulC_32f_C*R_Ctx`,
  dispatching on `dst.channels()` with explicit per-arm arrays.
- The scale denominator is the maximum positive representable value for the
  source type (Option B, resolved).

Update "How to add a new family" only if needed — Normalize is unusual (it wraps
one existing family's output); most new families will follow Resize/Mean/Convert.

### Step 2: Update `docs/roadmap.md`

**Update the F5.2 section** (lines 242–253). Replace the current text describing
Normalize codegen as deferred with the completed-status summary:

```
## F5.2 — Normalize codegen across the integer→f32 alphabet *(complete)*

**What:** Generalized the hand-written `Normalize<f32> for CudaImage<u8>` to
**every integer-source type for which NPP provides a non-rounding
`nppiConvert_*_Ctx → f32` symbol**, using a standalone generator that filters
the `nppiConvert_symbols.txt` fixture. No public API change. The scale constant
is the source type's maximum positive representable value (255 for u8,
65535 for u16, 32767 for i16). The hand-written impl was removed in favour of
the generated one; `convert_ops.rs` is retained as a documented placeholder.

**Key architectural decision:** Unlike the other four generated families,
Normalize does NOT use `FamilyDescriptor`/`generate_for_family()`. The
generator (`generate_normalize_impls`) is standalone; the convert step
delegates to the `ConvertTo` trait (not raw FFI); the three `MulC_32f` symbols
are hardcoded in the macro body. This avoids the `[c; $ch]` compilation bug
and the fragile `dual_type` hijacking.

**Supported source types:** `u8`, `u16`, `i16` (bounded by the F5.1 Convert
fixture; expands automatically on regeneration if new `→ f32` pairs appear).

**Committed artifacts added by F5.2:**
- `npp/src/normalize_macros.rs` — `impl_normalize_for!` macro
- `npp/src/normalize_generated.rs` — generated invocations (byte-identity guarded)
- `npp-codegen/src/gen_impls.rs` — `normalize_scale_denominator` helper +
  `generate_normalize_impls()` function
- `npp-codegen/examples/gen_normalize_impls.rs` — generator example
- `npp/tests/golden_normalize_16u32f.rs` — golden test for u16→f32
- `npp/tests/golden_normalize_16s32f.rs` — golden test for i16→f32

**Deleted:** hand-written `impl Normalize<f32> for CudaImage<u8>` from
`convert_ops.rs`.

**Point forward to F5.3:** The rounding-mode `ConvertTo` variants (20
`NppRoundMode` + 17 scaled functions) require a breaking API change and are
split to F5.3.
```

**Add a dedicated `## F5.3` section:**

```
## F5.3 — Rounding-mode ConvertTo (API change) *(deferred)*

**What:** Add the rounding-mode `nppiConvert_*` variants — the 20
`NppRoundMode` functions (shape `SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode`)
and the 17 scaled functions (shape `SRC+STEP, DST+STEP, SIZE,
MISC:NppRoundMode, CONST_SCALAR`) — which requires a **breaking public API
change** to the `ConvertTo` trait (a rounding-mode parameter, and a scale-factor
parameter for the scaled group).

**Why deferred:** F5.2 chose the additive Normalize half to stay shippable
without an API change; the rounding half is a self-contained API-breaking unit
~the size of F5.1.

**Dependencies:** F5.1 (dual-type codegen path) and F5.2 (Normalize precedent).
```

**Update the "Suggested rough sequencing" tree (line 393):** replace
`F5.2 (Normalize codegen + rounding Convert)` with
`F5.2 (Normalize codegen) ──> F5.3 (rounding-mode ConvertTo)`.

---

## Additional items (folded in from review and spar)

### A. No redundant dimension check (consistency)

The macro body does **not** add a leading dimension/channel validation step.
`self.convert(dst)?` already checks width/height/channels agreement inside the
generated `ConvertTo` impl (`convert_macros.rs:79–86`). Adding a redundant check
in the Normalize macro would duplicate the error wording and decorrelate from
the hand-written reference (`convert_ops.rs:42–44`).

### B. Defensive `Err` over `unreachable!()` for MulC `_` arm

The `_ =>` arm in the MulC dispatch returns
`Err(NppError::InvalidArgument(...))` rather than `unreachable!()`.
This matches the generated-macro convention (`convert_macros.rs:119`) and is
defensive in macro-generated unsafe-adjacent code. The arm IS genuinely
unreachable after a successful convert (both families share {1,3,4} channels).

### C. Float-literal precision tripwire (forward-looking)

`normalize_scale_denominator` returns u32/i32 denominators
(`4_294_967_295.0`, `2_147_483_647.0`). These are **not exactly f32
representable** and would trigger `clippy::excessive_precision` if emitted as
`…_f32` literals. They are **not emitted today** (only u8/u16/i16 have Convert
symbols, all exactly representable). Documented in the helper's doc comment;
the fix-when-it-happens is a `const` cast emission path.

### D. Golden expectations must be GPU-pinned, not fabricated

The scale is `(x as f32) * reciprocal`, not `(x as f32) / denom`. These differ
in the last ULP. All three goldens (existing u8 + two new) must be pinned from
actual GPU output, never hand-computed. The test doc blocks and code comments
reinforce this.

### E. Verified call-site safety

Grep confirms no production code outside `convert_ops.rs` directly references
the removed hand-written impl. Trait resolution routes through the generated
impl. The existing `golden_normalize.rs` test validates the resolution path at
build time.

### F. Definition of done (consolidated gate)

| Check | Command | When |
|-------|---------|------|
| Build | `cargo build` | Phase 4 |
| Build npp-rs | `cargo build -p npp-rs` | Phase 4 (MulC C4 gate) |
| Non-GPU test | `cargo test -p npp-rs --lib` | Phase 4 |
| Codegen test | `cargo test -p npp-codegen` | Phase 4 (byte-identity guard) |
| Lint | `cargo clippy -- -D warnings` | Phase 4 |
| Format | `cargo fmt --check` | Phase 4 |
| Coverage | `cargo tarpaulin` | Phase 4 (≥ 90%) |
| Regression | `convert_generated_is_byte_identical` | Phase 4 (must stay green) |
| Docs | `cargo doc --no-deps -p npp-rs -p npp-sys` | Phase 6 |
| GPU goldens | Manual on GPU host | Post-merge (known deferred follow-up) |

### G. GPU golden pinning is a known deferred follow-up

The two new golden tests ship with placeholder `EXPECTED` values and are
`#[cfg(feature = "gpu")]`-gated. They cannot be pinned in this work (no GPU
lane in CI). Pinning happens on a GPU host post-merge, exactly as the F5.1
convert goldens were pinned after merge (`6fa67a8`).

### H. `nppiMulC_32f_C4R_Ctx` symbol existence

Verified at build time in Phase 4 Step 1 (the `cargo build -p npp-rs` gate).
If absent, the C4 arm is removed from the macro body, restricting Normalize to
C1/C3 (matching the hand-written code's original scope). The C3 golden tests
exercise the array-dispatch path; C4 is validated structurally by the build,
not behaviourally.

---

## Risks & callouts (consolidated)

1. **MulC C4 gate (H)** — verified at build time; documented resolution if absent.
2. **Float precision tripwire (C)** — latent for future fixture growth; documented
   in helper doc.
3. **Golden pinning deferred (G)** — known pattern; no fabricating values.
4. **Coverage gate (F6)** — generated macros are FFI-dead under tarpaulin; match
   existing families (no annotation needed). Verified empirically in Phase 4.
5. **No API change in this plan** — `Normalize` and `ConvertTo` traits untouched.
   The API break lives entirely in F5.3.
6. **Bounded support set** — only `u8`, `u16`, `i16` currently; grows
   automatically on regeneration if the Convert fixture expands.
7. **Convert path untouched** — the existing `convert_generated_is_byte_identical`
   test is the regression guard; it must continue to pass.
