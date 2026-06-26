# Feature: F5.3 — Rounding-mode ConvertTo (new `ConvertRounded` trait)

## Goal (restated)

Add the **rounding-mode `nppiConvert_*` variants** (shape `SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode` — narrowing conversions like `f32→u8`) as a **new, non-breaking `ConvertRounded<Dst>` capability trait**, following the established fixture → classifier → `FamilyDescriptor` → macro → committed-`*_generated.rs` codegen pattern. The **scaled** variants (`+ CONST_SCALAR`) are explicitly **deferred to F5.4**.

> **[FIX I1] Symbol count is build-determined.** The shape survey lists `nppiConvert_ (20)` under the round-mode shape (`npp-shape-survey.md:833`), but that is the raw count across *all* type pairs. After `16f` exclusion and C1/C3/C4 variant grouping, the number of generated `(src,dst)` impl blocks is **build-determined, not a fixed target of 20.**

## Resolved decisions (binding for this plan)

| Decision | Resolution | Basis |
|----------|-----------|-------|
| API shape | **New trait `ConvertRounded<Dst>`** — additive, non-breaking | User; matches one-trait-per-capability house style |
| Scope | **Round-mode group only**; scaled → F5.4 | User; sidesteps the `shape.rs` `CONST_SCALAR` name-fragility |
| `RoundMode` enum | 3 modes: `Nearest`/`Financial`/`Zero` | NVIDIA `nppdefs.html` (verified) |
| Mode-unsupported handling | **Runtime** `check_status` (no probed table) | `error.rs:31` maps negative status (incl. `NPP_ROUND_MODE_NOT_SUPPORTED_ERROR`) → `NppError::Npp`; all 3 modes universally supported |
| Fixture strategy | **Separate** `nppiConvertRound_symbols.txt` + **separate** `classify_convert_round` | Round-mode symbols share `{src}{dst}_CxR` names with no-rounding symbols already in `nppiConvert_symbols.txt`; `classify_convert` is used by both `CONVERT_FAMILY` and Normalize against that fixture — sharing would collide |
| Generator dispatch | New `dual_type_round: bool` field **+ explicit call-site selector** | **[FIX B1]** `classify_convert` is *hardcoded* at `gen_impls.rs:193`/`:347`; a flag alone is inert |
| ROI support | **None** — owned-buffer only | Matches Convert/Normalize/Mean precedent (`roadmap.md:333-347`) |
| Golden organization | DRY — three per-mode tests share one `make_input` | User |
| Fixture coverage | Broad draft, build-pruned | User |
| Golden coverage depth | **Representative** (`f32→u8`), not exhaustive | **[FIX #4]** Per F5.1 precedent (`plan-f5.1.md:140`) |

## Overall strategy

Mirror F5.1 (the dual-type Convert codegen) almost exactly, plus one new safe enum (cloning the `ResizeInterpolation` precedent) and one new trait. Work is build-time-only and additive: **no existing trait, generated file, macro, or golden is modified.** The fixture ships as a **draft** validated on a CUDA/GPU host via build-resolution + a `syn` shape-check — the **load-bearing risk**, exactly as in F5.1.

> **Build-host confirmation required (do not assert from source):** the round-mode symbols, their bindgen names (`NppRoundMode_NPP_RND_*`, `nppiConvert_*_Ctx`), the parameter's bindgen *type*, and the `MISC:NppRoundMode` shape match are confirmable **only** by a green `cargo build -p npp-rs` + the shape-check test. `bindings.rs` is gitignored in `OUT_DIR`.

> **Forward-note for F5.4:** the scaled group's `CONST_SCALAR` role in `shape.rs:212-216` is detected by **parameter-name** heuristics (`Divisor`/`Value`/`Constant`/`ScaleFactor`). If the scale arg is named otherwise, `derive_shape` misclassifies it as `MISC:i32` and the shape-check rejects a valid symbol. F5.4 must extend `classify_param` first.

---

## Phase 1: `RoundMode` safe enum + trait + translator

Commit message: `feat(npp-rs): add RoundMode enum, ConvertRounded trait, and translator`

### Step 1: Add the `RoundMode` enum to `imageops.rs`

In `npp/src/imageops.rs`, add a public enum modelled exactly on `ResizeInterpolation` (`imageops.rs:7-19`):

```rust
/// Rounding modes for narrowing pixel conversions (`ConvertRounded`).
///
/// Controls how fractional source values are converted to integer
/// destination values. See NPP `NppRoundMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    /// Round to nearest; ties to **even** (`NPP_RND_NEAR`). E.g. 0.5→0, 1.5→2.
    Nearest,
    /// Round to nearest; ties **away from zero** (`NPP_RND_FINANCIAL`). E.g. 0.5→1, -1.5→-2.
    Financial,
    /// Truncate **toward zero** (`NPP_RND_ZERO`). E.g. 1.9→1, -2.5→-2.
    Zero,
}
```

Do **not** map the IEEE-754 alias enumerators — they share underlying values with the three canonical names and would be redundant.

### Step 2: Add the `ConvertRounded<Dst>` trait to `imageops.rs`

Modelled on `ConvertTo` (`imageops.rs:94-108`), with the `RoundMode` parameter:

```rust
/// Capability trait for narrowing cross-type conversion with explicit rounding.
///
/// Implemented only for `(src, dst)` pairs that NPP provides a rounding-mode
/// `nppiConvert_*` symbol for (narrowing conversions, e.g. `f32 → u8`).
/// Unsupported pairs simply have no impl — a compile-time error.
///
/// # Precondition
/// `self` and `dst` must not overlap; dimensions/channels must match (else
/// `NppError::InvalidArgument`). The CUDA device handle must outlive all
/// buffers (C7). This is an **owned-buffer** operation — no ROI sub-image
/// support (matching Convert/Normalize/Mean).
pub trait ConvertRounded<Dst: NppPixelType> {
    /// Convert `self` into `dst`, rounding fractional values per `mode`.
    ///
    /// # Errors
    /// `NppError::InvalidArgument` on dimension/channel/channel-count mismatch;
    /// `NppError::Npp` on NPP failure (including
    /// `NPP_ROUND_MODE_NOT_SUPPORTED_ERROR` if the pair rejects `mode`).
    fn convert_rounded(&self, dst: &mut CudaImage<Dst>, mode: RoundMode) -> Result<(), NppError>;
}
```

> **[FIX G2]** The owned-buffer-only scope is **deliberate**, not an oversight — it matches the Convert/Normalize/Mean precedent. Resize and SwapChannels got `pub(crate)` `*_into` ROI engines in F6.2; Convert-family ops did not.

### Step 3: Add the `round_mode` translator

Create `npp/src/convert_round_ops.rs`, modelled on `resize_ops.rs:8-16`:

```rust
use crate::imageops::RoundMode;

pub(crate) fn round_mode(mode: RoundMode) -> npp_sys::NppRoundMode {
    match mode {
        RoundMode::Nearest   => npp_sys::NppRoundMode_NPP_RND_NEAR,
        RoundMode::Financial => npp_sys::NppRoundMode_NPP_RND_FINANCIAL,
        RoundMode::Zero      => npp_sys::NppRoundMode_NPP_RND_ZERO,
    }
}
```

Add offline unit tests, one per variant, asserting the mapped value (mirroring `resize_ops.rs:42-80`).

> **[FIX I4] Coverage rationale:** `round_mode()` is **pure logic (no FFI)**, so unlike the generated FFI macros it **is** counted by `cargo tarpaulin` and **must** carry unit tests to hold ≥ 90%. Do **not** annotate it with `#[cfg(not(tarpaulin_include))]` — that is for GPU/FFI code only (AGENTS.md). Same applies to `classify_convert_round` (Phase 2).

> **Build-host note:** confirm the return type and constant names (`NppRoundMode`, `NppRoundMode_NPP_RND_*`) against `bindings.rs`. If bindgen emits the round-mode param as `i32` rather than a typed enum, change the return type to `i32` and append `as i32` (matching `interpolation_mode` at `resize_ops.rs:8`).

### Step 4: Wire `convert_round_ops` into `lib.rs`

Add `pub mod convert_round_ops;` to `npp/src/lib.rs` in alphabetical position (after `convert_ops`, before `cuda`), with a doc comment. Do **not** yet add the macro/generated modules (Phase 3). Verify `cargo build -p npp-rs` is green.

---

## Phase 2: `classify_convert_round` + `CONVERT_ROUND_FAMILY` in npp-codegen

Commit message: `feat(npp-codegen): add round-mode Convert classifier and family descriptor`

### Step 1: Add `classify_convert_round` to `classify.rs`

In `npp-codegen/src/classify.rs`, add `pub fn classify_convert_round(symbols: &[&str]) -> Vec<ClassifiedSymbol>`. Near-copy of `classify_convert` (`classify.rs:193-286`): **same two-token split, same `C1R/C3R/C4R` + `_Ctx` acceptance, same prefer-`_Ctx` dedup, same `op = "Convert"` and `dst_type_token = Some(dst)`.** Doc comment must state: this classifies the **rounding-mode** Convert group; the symbol *names* are identical to no-rounding Convert, so **the shape-check (Step 4 / Phase 4) is what guarantees** a fixture symbol truly has the round-mode shape — name classification alone cannot distinguish them.

### Step 2: Unit-test `classify_convert_round`

Extend `#[cfg(test)] mod tests`, mirroring the `classify_convert` tests (`classify.rs:566-643`):
- Accept `nppiConvert_32f8u_C1R` → `type_token=="32f"`, `dst_type_token==Some("8u")`, `channels==1`.
- Prefer `_Ctx`: given both `nppiConvert_32f8u_C1R` and `_C1R_Ctx`, return one result with `variant` ending `_Ctx`.
- Reject single-token `nppiConvert_8u_C1R`.
- Reject `16f`-bearing (`nppiConvert_32f16f_C1R`).
- Reject non-standard variant (`nppiConvert_32f8u_C4C3R`).

### Step 3: Add `dual_type_round` field + `CONVERT_ROUND_FAMILY` descriptor

**[FIX B1] — two distinct edits in `gen_impls.rs`:**

**(a) Add the field.** Add `pub dual_type_round: bool` to `FamilyDescriptor` (struct at `gen_impls.rs:15-45`), documented: "When true, the dual-type generator path uses `classify_convert_round` instead of `classify_convert`. Default `false`." **Set `dual_type_round: false` on all four existing descriptors** (`RESIZE_FAMILY`, `SWAP_CHANNELS_FAMILY`, `MEAN_FAMILY`, `CONVERT_FAMILY`) so the crate compiles.

**(b) Add the call-site selector.** In `generate_for_family`, the dual-type branch **hardcodes** `classify_convert` at **`gen_impls.rs:193`**. Replace that single line with:

```rust
let classified = if family.dual_type_round {
    classify_convert_round(&symbol_refs)
} else {
    classify_convert(&symbol_refs)
};
```

(Import `classify_convert_round` alongside the existing `use crate::classify::{classify, classify_convert};` at `gen_impls.rs:8`.) The rest of the dual-type emit block (`:195-241`) is **unchanged** — the emit shape is byte-identical; the *macro*, not the generator, adds the round-mode argument.

**(c) Add the descriptor.** `pub static CONVERT_ROUND_FAMILY: FamilyDescriptor` modelled on `CONVERT_FAMILY` (`gen_impls.rs:114-132`):
- `npp_prefix: "nppiConvert_"`, `accepted_channels: &[1,3,4]`, `custom_variants: &[]`
- `macro_name`/`rust_macro_path: "impl_convert_rounded_for"`
- `expected_shape: "SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode"`
- `skip_16f: true`, `dual_type: true`, **`dual_type_round: true`**, `get_buffer_host_size_prefix: None`, `engine_fn_prefix: None`
- `use_statements` (for the **generated file**): `use crate::error::{check_status, NppError};`, `use crate::image::CudaImage;`, `use crate::imageops::{ConvertRounded, RoundMode};` **[FIX G1]**, `use crate::impl_convert_rounded_for;`, `use npp_sys::NppiSize;`

> **[FIX #2 — unverified shape-string link]** `expected_shape` assumes bindgen types the parameter as the `NppRoundMode` enum, so `shape.rs:227`'s catch-all emits `MISC:NppRoundMode`. **If bindgen flattens it to `int`, `derive_shape` emits `MISC:i32` instead** and `expected_shape` must be adjusted to match. The `convert_round_syn_shape_check` test (Step 4) prints the actual derived shape on mismatch — **treat its output as authoritative and adjust `expected_shape` to whatever it reports**, rather than treating this string as gospel.

### Step 4: Add generator unit tests + shape-check

**[FIX B1] — `validate_symbols_against_bindings` also hardcodes the classifier** at **`gen_impls.rs:347`**. Apply the same selector there:

```rust
let classified: Vec<ClassifiedSymbol> = if family.dual_type_round {
    crate::classify::classify_convert_round(&symbol_refs)
} else if family.dual_type {
    crate::classify::classify_convert(&symbol_refs)
} else {
    classify(&symbol_refs, family.npp_prefix, family.accepted_channels, family.custom_variants)
};
```

Then in `#[cfg(test)]`:
- `generate_convert_round_from_fixture`: reads `nppiConvertRound_symbols.txt`, runs `generate_for_family(&CONVERT_ROUND_FAMILY, …)`, asserts output contains `impl_convert_rounded_for!(f32, u8, "32f", "8u", {`, a `_Ctx` symbol path, and no `16f`/`f16`.
- `convert_round_syn_shape_check`: mirrors `convert_syn_shape_check` (`gen_impls.rs:906`). Validates every fixture symbol has the expected round-mode shape; skips with a notice if `bindings.rs` absent.

---

## Phase 3: `impl_convert_rounded_for!` macro + committed placeholder

Commit message: `feat(npp-rs): add impl_convert_rounded_for macro and wire generated module`

### Step 1: Write the `impl_convert_rounded_for!` macro

Create `npp/src/convert_round_macros.rs` defining `#[macro_export] macro_rules! impl_convert_rounded_for!` with signature `($src_ty:ty, $dst_ty:ty, $src_token:expr, $dst_token:expr, { $($ch:literal => $sym:path),+ $(,)? })`, expanding to `impl ConvertRounded<$dst_ty> for CudaImage<$src_ty>`. A **near-copy of `impl_convert_for!`** (`convert_macros.rs:38-133`) with exactly two differences:

1. Method is `fn convert_rounded(&self, dst: &mut CudaImage<$dst_ty>, mode: RoundMode) -> Result<(), NppError>`.
2. Each `$sym` call appends **one argument** — `$crate::convert_round_ops::round_mode(mode)` — **between `nppi_size` and `self.ctx.raw_ctx()`**:

```rust
$ch => $sym(
    src_ptr as *const _, src_step_bytes,
    dst_ptr as *mut _,   dst_step_bytes,
    nppi_size,
    $crate::convert_round_ops::round_mode(mode),
    self.ctx.raw_ctx(),
),
```

Everything else identical: dimension/channel agreement check, dual-type `src_step_bytes`/`dst_step_bytes`, pointers via `DevicePtr::device_ptr(&self.buf)` / `DevicePtrMut::device_ptr_mut(&mut dst.buf)` (`buf` is `pub(crate)`, confirmed `image.rs:104`), the `_ =>` `InvalidArgument` arm naming both tokens, `check_status(status)`, `#[allow(clippy::macro_metavars_in_unsafe)]`. **Do not** add `#[cfg(not(tarpaulin_include))]`.

> **[FIX G1]** The macro names `RoundMode` in the method signature; `mode` flows to the fully-qualified `$crate::convert_round_ops::round_mode(...)`. `RoundMode` must be **in scope at the call site** — imported in `convert_round_generated.rs` (handled by Phase 2 Step 3c `use_statements`). The macro file itself needs no `use` (it is `#[macro_export]`).

`#[macro_export]`-level doc comment documents: args, the round-mode parameter, **narrowing direction** (`src_step_bytes > dst_step_bytes` — opposite of F5.1's widening), non-overlap precondition, C7 invariant, owned-buffer-only (no `img_index`, per `convert_macros.rs:59-62`).

### Step 2: Create committed `convert_round_generated.rs` placeholder

Create `npp/src/convert_round_generated.rs` with the standard generated header (`//! GENERATED — re-run \`cargo run --example gen_convert_round_impls\` …` + the committed-not-gitignored line), use statements **including `use crate::imageops::{ConvertRounded, RoundMode};`** **[FIX G1]**, and **one minimal invocation**:

```rust
impl_convert_rounded_for!(f32, u8, "32f", "8u", {
        1 => npp_sys::nppiConvert_32f8u_C1R_Ctx,
        3 => npp_sys::nppiConvert_32f8u_C3R_Ctx,
});
```

Copy exact formatting (8-space arm indent, trailing `});`) from `convert_generated.rs`. **Do not** add the byte-identity test yet (placeholder ≠ full output; lands in Phase 4 to keep every commit green — `plan-f5.1.md:118`).

### Step 3: Wire modules into `lib.rs`

Add (alphabetical, with doc comments): `pub mod convert_round_generated;` and `pub mod convert_round_macros;`. Update the crate-level doc (`lib.rs:12-14`) to mention `ConvertRounded` for narrowing conversions. Verify `cargo build -p npp-rs` and `cargo test -p npp-rs --lib` green.

---

## Phase 4: Fixture, generator example, regenerate, byte-identity guard, verification

Commit message: `test(npp-rs): add round-mode Convert fixture, regenerate, byte-identity guard`

### Step 1: Create the generator example

**[FIX I2] — this must come BEFORE the regenerate command in Step 3.** Create `npp-codegen/examples/gen_convert_round_impls.rs`, cloning `gen_convert_impls.rs` (`gen_convert_impls.rs:1-30`): read `nppiConvertRound_symbols.txt` via `read_fixture`, call `generate_for_family(&CONVERT_ROUND_FAMILY, &symbols)`, write to `../npp/src/convert_round_generated.rs`. Top doc: `cargo run --example gen_convert_round_impls`.

### Step 2: Create the draft fixture

Create `npp-codegen/tests/fixtures/nppiConvertRound_symbols.txt`. Leading comment in `nppiConvert_symbols.txt` style: **rounding-mode group only**; scaled → **F5.4**; `16f` excluded; regenerate on CUDA bump; **DRAFT — validated against `bindings.rs` in Step 3; do not invent symbols.** Populate **broadly**, **both bare and `_Ctx` for each** (the `_Ctx` listing is load-bearing for the stream pivot). Candidates (build will prune): `32f→{8u,8s,16u,16s,32s}`, `16s→{8u,8s}`, `16u→8u`, `32s→{8u,8s,16u,16s}`, each ×{C1R,C3R,C4R} ×{bare,_Ctx}.

### Step 3: Regenerate + build (symbol-resolution gate)

```bash
nix develop . --command cargo run -p npp-codegen --example gen_convert_round_impls
nix develop . --command cargo build -p npp-rs
```

**This build is the symbol-existence gate** — any non-existent fixture symbol surfaces as an unresolved-path error. Correct the fixture against `bindings.rs` and regenerate until green. Then run `convert_round_syn_shape_check` on this bindings-available host to confirm shapes (**and adjust `expected_shape` if the test reports a different derived shape — see [FIX #2]**).

### Step 4: Add the byte-identity guard

In `gen_impls.rs` `#[cfg(test)]`, add `convert_round_generated_is_byte_identical`, mirroring `convert_generated_is_byte_identical` (`gen_impls.rs:644-682`). Lands in the **same commit** as the regenerated file.

### Step 5: Verification gates (Nix dev shell)

All green: `cargo build`, `cargo build -p npp-rs`, `cargo test -p npp-rs --lib`, `cargo test -p npp-codegen`, `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo tarpaulin` (≥ 90%).

> **[FIX I3]** Existing byte-identity guards number **five** (resize, convert, swap_channels, mean, normalize — `gen_impls.rs:564,604,644,684,724`); all five must stay green. After this phase: **six**.

> **[FIX I4]** Generated round-mode impls are FFI-dead under non-GPU tarpaulin like the other families. `round_mode()`/`classify_convert_round()` are pure logic — their Phase 1/2 unit tests keep coverage up. Investigate before annotating if coverage drops.

---

## Phase 5: Golden tests (GPU-gated, pinned post-merge)

Commit message: `test(npp-rs): add round-mode Convert golden tests`

> **[FIX #4] Golden coverage is representative, not exhaustive.** These goldens cover **`f32→u8`** only. If the build reveals other round-mode pairs (e.g. `f32→16u`), those impls ship covered by the **byte-identity guard + shared macro path** but **no dedicated pixel golden** — matching F5.1, which shipped a representative subset (`plan-f5.1.md:140`), not one golden per pair. This is a deliberate coverage decision.

### Step 1: Standalone per-mode golden — `f32 → u8` (DRY, three tests, one input)

Create `npp/tests/golden_convert_round_32f8u.rs` following `golden_convert_8u16u.rs` structure. One shared `make_input` producing **fractional `f32` values where the three modes diverge**, plus an integer anchor:

```rust
// Chosen so Nearest/Financial/Zero produce DIFFERENT u8 outputs.
// 0.5 (Near→0, Fin→1, Zero→0), 2.5 (Near→2, Fin→3, Zero→2),
// 1.9 (Near/Fin→2, Zero→1), 5.0 (all→5, mode-invariant anchor).
let pattern = [0.5_f32, 2.5, 1.9, 5.0, 0.4, 3.5];
```

Write **three `#[test]` functions** sharing that one `make_input` — one per `RoundMode` — each calling `src.convert_rounded(&mut dst, mode)` with its **own** placeholder (`const EXPECTED_NEAR/_FIN/_ZERO: &[u8] = &[]`). Doc block: pin each from a real GPU run; **never hand-compute** (`plan-f5.2.md:767`); identical pins across modes would indicate the `mode` parameter is not plumbed.

### Step 2: Chained `_Ctx` golden (drop-off detector)

Create `npp/tests/golden_convert_round_chained.rs` modelled on `golden_chained_ctx.rs`. On one shared `StreamContext`, chain `resize` (`_Ctx`) → `convert_rounded(f32→u8, Nearest)` (`_Ctx`) with a single host-fenced readback (`TryFrom<&CudaImage<u8>> for Vec<u8>`, confirmed `image.rs:460`).

> **[FIX G3]** This test is a **`_Ctx`-plumbing regression guard only**, **not** a round-mode-semantics test. The intermediate (resize output) `f32` values depend on interpolation and are **not** the clean `0.5/2.5` of Step 1, so the chained `EXPECTED` is **not** hand-reasoned — it is purely pinned from GPU output. Round-mode semantics are owned exclusively by Step 1.

### Step 3: GPU pin (post-merge follow-up)

Per F5.2 convention (`plan-f5.2.md:796`), goldens ship with empty `EXPECTED` and are pinned on a GPU host post-merge. Document the pin command in each file's header.

---

## Phase 6: Documentation reconciliation

Commit message: `docs: reconcile F5.3 round-mode ConvertTo, add F5.4 entry`

### Step 1: Update `docs/codegen-architecture.md`

Add a `ConvertRounded` row to "Families implemented" (`codegen-architecture.md:110-116`): prefix `nppiConvert_`, C1/C3/C4, shape `SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode`, dual-type Yes. Add a short subsection: the separate fixture/classifier (name-collision rationale), the `dual_type_round` selector, and that the macro (not generator) injects the round-mode argument.

### Step 2: Update `docs/roadmap.md`, `lib.rs`, `gen_impls.rs` doc strings

- **Correct the "breaking API change" framing.** `roadmap.md:290-297` calls F5.3 a "breaking public API change to the `ConvertTo` trait." Replace with: F5.3 is **additive** — a new `ConvertRounded<Dst>` trait, non-breaking. Mark F5.3 **complete** with a committed-artifacts list.
- **Add a first-class `## F5.4` entry**: the scaled `nppiConvert_*` variants (shape `…, MISC:NppRoundMode, CONST_SCALAR`), requiring a scale-factor parameter **and** the `shape.rs` `CONST_SCALAR` extension. Dependencies: F5.3.
- Update the sequencing tree (`roadmap.md:469`) `F5.3 ──> F5.4`.
- Fix stale "deferred to F5.3 / F5.2" doc strings in `lib.rs:14,40` and `gen_impls.rs:113`.

---

## Risks & callouts (consolidated)

1. **Fixture accuracy (load-bearing)** — guarded twice: Phase 4 Step 3 build-resolution + `convert_round_syn_shape_check`. Fixture is a **draft**; shrinking it is the abort path. Confirmable only on a CUDA host.
2. **Generator dispatch [FIX B1]** — the `dual_type_round` selector must be applied at **both** hardcoded `classify_convert` sites (`gen_impls.rs:193` and `:347`); a field without the call-site edits is inert. The five existing `*_is_byte_identical` guards are the tripwire.
3. **`_Ctx` drop-off** — fixture lists both forms + prefer-`_Ctx` dedup + the Phase 5 Step 2 chained golden.
4. **Silent round-mode pass** — divergent fractional inputs + one pinned golden *per mode* + integer anchor (Phase 5 Step 1).
5. **Bindgen name/type unknowns [FIX #2]** — `NppRoundMode_NPP_RND_*` names, and whether the param is a typed enum or `i32` (which decides both the translator return type and `expected_shape`); resolved at the Phase 1 Step 3 build and the Phase 4 Step 3 shape-check.
6. **Coverage [FIX I4]** — generated impls FFI-dead like existing families; `round_mode`/`classify_convert_round` are pure logic and rely on their unit tests. No `#[cfg(not(tarpaulin_include))]` on them.
7. **F5.4 shape.rs debt** — flagged forward; not in F5.3 scope.

---

## Definition of done

| Check | Command | Phase |
|-------|---------|-------|
| Build | `cargo build` | 4 |
| Build npp-rs | `cargo build -p npp-rs` | 1, 3, 4 (symbol gate) |
| Non-GPU test | `cargo test -p npp-rs --lib` | 1, 3, 4 |
| Codegen test (6 byte-identity + classifier + shape-check) | `cargo test -p npp-codegen` | 4 |
| Lint / Format | `cargo clippy -- -D warnings`, `cargo fmt --check` | 4 |
| Coverage ≥ 90% | `cargo tarpaulin` | 4 |
| Docs | `cargo doc --no-deps -p npp-rs -p npp-sys` | 6 |
| GPU goldens (4 tests: 3 modes + chained) | manual on GPU host | post-merge |
