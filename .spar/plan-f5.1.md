# Feature: F5.1 — Cross-type ConvertTo codegen across the NppPixelType alphabet

## Goal (restated)

Generalize the F5 hand-written `u8→f32` `ConvertTo` implementation to **every `(src, dst)` pixel-type pair that NPP provides a non-rounding `nppiConvert_*_Ctx` symbol for**, using the established fixture → classifier → generator → macro → committed-`*_generated.rs` codegen pattern. The hand-written `Normalize<f32> for CudaImage<u8>` stays. No public API change. Coverage stays ≥ 90% and the single-type generators stay byte-identical.

> The **exact set** of `(src, dst)` pairs that receive impls is **determined by the CUDA-host build**, not by this plan: a pair is included iff its `nppiConvert_*_Ctx` symbol exists and fits the no-rounding `SRC+STEP, DST+STEP, SIZE` shape. "Generalize to the alphabet" therefore means "every such pair NPP actually provides," not a guarantee of any specific pair.

## Overall strategy

The one genuinely new mechanism is **dual-type classification**: a Convert symbol carries two type tokens (`8u32f` = src+dst). We add a dual-type path to `npp-codegen` (`classify_convert` + a `dst_type_token` field + a `dual_type` flag on `FamilyDescriptor`), a `CONVERT_FAMILY` descriptor, a `gen_convert_impls` generator, an `impl_convert_for!($src_ty, $dst_ty, …)` macro, and a committed `convert_generated.rs`. The hand-written `ConvertTo` is removed in favour of the generated impl; `convert_ops.rs` is slimmed to host only `Normalize`. The work is build-time-only and mirrors F2 exactly.

---

## Phase 1: Dual-type classification in npp-codegen

Commit message: feat(npp-codegen): add dual-type symbol classification for Convert family

### Step 1: Add a dual-type ClassifiedSymbol field and a classify_convert function

In `npp-codegen/src/classify.rs`:

- Add `pub dst_type_token: Option<String>` to the `ClassifiedSymbol` struct (struct definition at line 5), documented as "the destination NPP element-type token for dual-type families (e.g. Convert); `None` for single-type families."
- **[B1]** Add `dst_type_token: None` to **both** existing `ClassifiedSymbol { … }` construction sites so the crate still compiles: (a) the second-pass loop in `classify()` at **`classify.rs:123–129`**, and (b) any construction in `gen_impls.rs`. The implementer must grep for `ClassifiedSymbol {` across the crate and confirm every literal is updated; the `Option<String>` type makes `None` the trivial default.
- Add `pub fn classify_convert(symbols: &[&str]) -> Vec<ClassifiedSymbol>`. Parse symbols of the form `nppiConvert_{SRC}{DST}_{VARIANT}`. Accept only standard variants `C1R`/`C3R`/`C4R` and their `_Ctx` forms.
- **[B2]** `classify_convert` must store `variant` **including** the `_Ctx` suffix when the `_Ctx` form is chosen (e.g. store `"C3R_Ctx"`, not `"C3R"`), identical to how `classify()` does it at `classify.rs:113–121` (prefer-`_Ctx` selection sets *both* `raw` and the suffixed `variant`). This is what makes the generated symbol path resolve to the `_Ctx` stream-context variant; storing the bare variant would silently pivot Convert off the F8 stream path.
- Set `op = "Convert"`, `type_token = src`, `dst_type_token = Some(dst)`, `channels`/`variant` as usual.
- **[G1]** Specify the two-token split algorithm precisely (the token segment is **not** prefix-free, so a naive scan is wrong): *Let `TOKENS = ["8u","8s","16u","16s","32u","32s","32f","64f"]` (16f excluded). For the segment between `nppiConvert_` and the first `_`, iterate every split point `k`; a split is valid iff `segment[..k] ∈ TOKENS` **and** `segment[k..] ∈ TOKENS`. Collect all valid splits. If exactly one valid split exists, use it. If **zero** valid splits exist, reject the symbol (skip it). If **two or more** valid splits exist, this is a classifier ambiguity bug — `debug_assert!`/panic in the generator (build-time only, never runtime) so it surfaces immediately rather than silently picking one.*
- Doc comment must explain the two-token split strategy, the zero/one/many handling, and why `16f` is excluded (safe layer disables the `half` crate).

### Step 2: Unit-test the dual-type classifier

Extend the `#[cfg(test)] mod tests` in `classify.rs`. Assert exact field values (style of `assert_classify_one_resize`):

- Accept `nppiConvert_8u32f_C1R` → `type_token=="8u"`, `dst_type_token==Some("32f")`, `channels==1`, `op=="Convert"`.
- Accept `nppiConvert_8u16u_C3R` (integer widening) → src `8u`, dst `16u`, channels 3.
- Accept `nppiConvert_16u32f_C4R` → src `16u`, dst `32f`, channels 4.
- **[B2]** Prefer `_Ctx`: given both `nppiConvert_8u32f_C1R` and `nppiConvert_8u32f_C1R_Ctx`, return exactly one result whose `variant` **ends with `_Ctx`** and whose `raw` is the `_Ctx` symbol.
- Reject `nppiConvert_8u_C1R` (single token → not dual-type; zero valid splits).
- Reject `16f`-bearing symbols `nppiConvert_32f16f_C1R` and `nppiConvert_16f32f_C1R` (both yield no results).
- Reject non-standard variant `nppiConvert_8u32f_C4C3R`.
- **[G1]** Add a split-robustness test: confirm a symbol whose segment *could* look mis-splittable resolves to the single correct `(src,dst)` pair (e.g. `nppiConvert_16u16u_C1R` → `("16u","16u")`, not some other split), locking the "exactly one valid split" contract.

---

## Phase 2: Dual-type generation in npp-codegen

Commit message: feat(npp-codegen): emit dual-type impl_convert_for invocations

### Step 1: Add a dual_type flag to FamilyDescriptor and the CONVERT_FAMILY descriptor

In `npp-codegen/src/gen_impls.rs`:

- Add `pub dual_type: bool` to `FamilyDescriptor`, documented as "When true, the family carries two type tokens (src+dst); the generator uses `classify_convert` and emits `impl_*_for!($src_ty, $dst_ty, \"$src_tok\", \"$dst_tok\", { … })`." Set `dual_type: false` on `RESIZE_FAMILY`, `SWAP_CHANNELS_FAMILY`, `MEAN_FAMILY`.
- Add `pub static CONVERT_FAMILY: FamilyDescriptor`: `npp_prefix: "nppiConvert_"`, `accepted_channels: &[1,3,4]`, `custom_variants: &[]`, `macro_name: "impl_convert_for"`, `rust_macro_path: "impl_convert_for"`, `expected_shape: "SRC+STEP, DST+STEP, SIZE"`, `skip_16f: true`, `get_buffer_host_size_prefix: None`, `dual_type: true`, `use_statements` = exactly: `use crate::error::{check_status, NppError};`, `use crate::image::CudaImage;`, `use crate::imageops::ConvertTo;`, `use crate::impl_convert_for;`, `use npp_sys::NppiSize;`.
- **[F5.2]** Add a doc comment on `CONVERT_FAMILY` stating it covers the **no-rounding** shape (`SRC+STEP, DST+STEP, SIZE`) **only**; rounding-mode Convert variants (`NppRoundMode`) are **deferred to F5.2**.

### Step 2: Branch generate_for_family on dual_type to emit two-type invocations

In `gen_impls.rs`, modify `generate_for_family()` so when `family.dual_type` is true it:

- Calls `classify_convert(&symbol_refs)` instead of `classify(...)`.
- Groups by `(type_token, dst_type_token)` pair, each group collecting `channel => variant` arms (reuse the `BTreeMap<u8,String>` arm structure keyed under the `(src,dst)` pair).
- **[G1/skip_16f]** Skips a group if **either** the src or dst token fails `npp_type_to_rust`, and applies the `skip_16f` check to **both** tokens.
- Emits one block per `(src,dst)` group:
  ```
  impl_convert_for!(<src_rust_ty>, <dst_rust_ty>, "<src_token>", "<dst_token>", {
          1 => npp_sys::nppiConvert_<src><dst>_C1R_Ctx,
          3 => npp_sys::nppiConvert_<src><dst>_C3R_Ctx,
  });
  ```
- **[B2]** Build each symbol path as `npp_sys::nppiConvert_{src}{dst}_{variant}` where **`{variant}` is the stored variant string that already includes `_Ctx`** (matching the single-type path at `gen_impls.rs:184`). Do **not** append `_Ctx` separately — it is already in `variant` per Phase 1 Step 1.
- **[G2]** The single-type `else` branch must remain **byte-for-byte unchanged**; the existing `resize_generated_is_byte_identical` test is the guard.
- Preserve the header-comment + `use`-statement emission and the blank-line-between-blocks formatting. The header "re-run" hint reads `gen_convert_impls`.

### Step 3: Add the convert fixture and the gen_convert_impls example

- Create `npp-codegen/tests/fixtures/nppiConvert_symbols.txt`: the **no-rounding** Convert symbols (shape `SRC+STEP, DST+STEP, SIZE`) from CUDA 12.9, one per line, leading comment block in the style of `nppiMean_symbols.txt`. **[F5.2]** Header comment must state: no-rounding group only; rounding-mode variants tracked under F5.2; `16f` excluded; regenerate on CUDA bump. Populate with the integer-widening and integer↔float non-rounding pairs NPP ships (e.g. `nppiConvert_8u16u_C1R`, `nppiConvert_8u32f_C1R/C3R/C4R`, `nppiConvert_16u32f_C1R/C3R/C4R`, …). **Treat this list as a draft** — it is validated against `bindings.rs` in Phase 4 Step 1; do not invent symbols.
- Create `npp-codegen/examples/gen_convert_impls.rs`, modelled on `gen_mean_impls.rs`: read the fixture via `read_fixture`, call `generate_for_family(&CONVERT_FAMILY, &symbols)`, write to `../npp/src/convert_generated.rs`. Top doc comment: run command `cargo run --example gen_convert_impls`.

### Step 4: Add generator unit tests for the Convert family

In `gen_impls.rs` `#[cfg(test)]`:

- `generate_convert_from_fixture`: reads `nppiConvert_symbols.txt`, runs `generate_for_family(&CONVERT_FAMILY, …)`, asserts the output is non-empty, contains `impl_convert_for!(u8, f32, "8u", "32f", {`, contains an integer-widening invocation such as `impl_convert_for!(u8, u16, "8u", "16u", {`, **[B2]** contains a `_Ctx` symbol path (e.g. `npp_sys::nppiConvert_8u32f_C3R_Ctx`), and does **not** contain any `16f`/`f16` token. Style of `generate_resize_from_fixture`.

- **[O1]** `convert_syn_shape_check`: mirrors `resize_syn_shape_check`/`swap_channels_syn_shape_check` (`gen_impls.rs:404,429`) — locate `bindings.rs` via `find_bindings_rs()`, and if present, validate that every Convert fixture symbol has shape `SRC+STEP, DST+STEP, SIZE`. **This requires extending `validate_symbols_against_bindings`** (`gen_impls.rs:218`, which currently calls `classify(...)` at line 243) with a `dual_type` branch that calls `classify_convert` instead — add that branch as part of this step. If `bindings.rs` is absent, the test prints a skip notice (same as the existing checks). This is the mechanical guard for the #1 risk (fixture accuracy).

---

## Phase 3: The runtime impl_convert_for! macro

Commit message: feat(npp-rs): add impl_convert_for macro for cross-type conversion

### Step 1: Write the dual-type convert macro

Create `npp/src/convert_macros.rs` defining `#[macro_export] macro_rules! impl_convert_for!` with signature `($src_ty:ty, $dst_ty:ty, $src_token:expr, $dst_token:expr, { $($ch:literal => $sym:path),+ $(,)? })`, expanding to `impl ConvertTo<$dst_ty> for CudaImage<$src_ty>` with `fn convert(&self, dst: &mut CudaImage<$dst_ty>) -> Result<(), NppError>`. Body replicates the current hand-written `convert` semantics:

- **[O2/ratified]** Do **not** add `#[cfg(not(tarpaulin_include))]` to the `impl_convert_for!` expansion. The three existing generated macros (`impl_resize_for!`, `impl_mean_for!`, `impl_swap_channels_for!`) carry no such annotation, yet the project holds ≥ 90% coverage: their generated bodies call `npp_sys` FFI and are dead code in the non-GPU tarpaulin lane, contributing no uncovered lines. `impl_convert_for!` is structurally identical, so it must **match the existing macros** (no annotation) rather than diverge. Phase 4 Step 4 confirms the gate still holds.
- Validate dimension/channel agreement vs `dst`; `NppError::InvalidArgument` on mismatch (reuse existing wording).
- Compute `src_step_bytes = self.layout.height_stride * size_of::<$src_ty>()` and `dst_step_bytes = dst.layout.height_stride * size_of::<$dst_ty>()` as `i32` (the differing element sizes are the crux of cross-type stepping).
- Extract raw pointers via `cudarc::driver::DevicePtr::device_ptr(&self.buf)` / `DevicePtrMut::device_ptr_mut(&mut dst.buf)`, each offset by its own `layout.img_index`, cast to `*const $src_ty` / `*mut $dst_ty`.
- `NppiSize` from `dst` dimensions.
- `match self.channels()` over `$ch => $sym` arms, each calling `$sym(src_ptr as *const _, src_step_bytes, dst_ptr as *mut _, dst_step_bytes, nppi_size, self.ctx.raw_ctx())`. Unmatched arm returns `NppError::InvalidArgument` naming the channel count and both tokens (`$src_token` → `$dst_token`).
- End with `check_status(status)`.
- **[F5.2]** `#[macro_export]`-level doc comment documents args, expansion, nStep-bytes-per-element-type, sub-image `img_index` offset, non-overlap (neighbourhood-gather) precondition, C7 device-lifetime invariant — **and** a line stating this implements unscaled, non-rounding conversion; scaled/rounding conversion is **F5.2**. Style of `impl_resize_for!`.

### Step 2: Wire the macro and generated module into lib.rs; remove the hand-written ConvertTo

- In `npp/src/lib.rs`: add `pub mod convert_macros;` (doc: "Macro to generate cross-type `impl ConvertTo` for image types.") and `pub mod convert_generated;` (doc: "Generated `impl ConvertTo` for all NPP-supported (src,dst) pairs (committed artifact)."). Keep `pub mod convert_ops;`.
- In `npp/src/convert_ops.rs`: **remove** `impl ConvertTo<f32> for CudaImage<u8>`. **Keep** `impl Normalize<f32> for CudaImage<u8>`. **[F5.2]** Update the module doc to state it now hosts only the hand-written `Normalize` slice (ConvertTo is generated in `convert_generated.rs`), and that generalizing `Normalize` across the alphabet is **deferred to F5.2**. Fix `use` imports so the file compiles with only `Normalize` (drop unused `ConvertTo` if unused; retain `check_status`, `CudaImage`, `Normalize`, cudarc ptr traits, `NppiSize`, `size_of`). Note: `normalize` calls `self.convert(dst)?`, which now binds to the **generated** `ConvertTo<f32> for CudaImage<u8>` — behaviourally identical, so `Normalize` works unchanged.
- **[G3]** Confirm in the step that `npp/tests/golden_convert.rs` needs **no changes**: it imports the `ConvertTo` trait and calls `src.convert(&mut dst)`, which still resolves via the trait + generated impl. Do not edit the golden.

### Step 3: Create a committed convert_generated.rs placeholder

- Create `npp/src/convert_generated.rs` with the standard generated header comment (`//! GENERATED — re-run \`cargo run --example gen_convert_impls\` on CUDA bump.` + the "committed, not gitignored" line), the five `use` statements, and at minimum `impl_convert_for!(u8, f32, "8u", "32f", { 1 => npp_sys::nppiConvert_8u32f_C1R_Ctx, 3 => npp_sys::nppiConvert_8u32f_C3R_Ctx })` so the crate and the retained `Normalize` compile. Copy the exact formatting (8-space arm indentation, trailing `});`) from `mean_generated.rs`.

- **[G2/deferred to Phase 4]** Do **not** add the byte-identity test in this phase. The placeholder cannot equal the generator's full output, so the test would be red at this commit — violating "never commit a known-failing test." The placeholder's only job is a green compile for the retained `Normalize`. (The byte-identity guard is introduced in Phase 4 Step 1, after regeneration, when it can pass.)

---

## Phase 4: Regenerate, verify, and pin goldens

Commit message: test(npp-rs): regenerate convert_generated and pin representative goldens

### Step 1: Regenerate convert_generated.rs and validate the fixture against real NPP symbols

Run `nix develop . --command cargo run -p npp-codegen --example gen_convert_impls` to overwrite `npp/src/convert_generated.rs` with the full invocation set. Then `nix develop . --command cargo build` and `nix develop . --command cargo build -p npp-rs` to confirm **every generated symbol path resolves** against bindgen output. **Any non-existent fixture symbol surfaces here as an unresolved-path error** — correct the fixture to match `bindings.rs` and regenerate until the build is green. **[O1]** Run the `convert_syn_shape_check` test from Phase 2 Step 4 on this GPU/bindings-available host to mechanically confirm shapes.

- **[G2]** Now that `convert_generated.rs` holds its full regenerated content, **add** the `convert_generated_is_byte_identical` test to `npp-codegen/src/gen_impls.rs` `#[cfg(test)]`, mirroring `resize_generated_is_byte_identical` (lines 307–345): generate from `nppiConvert_symbols.txt` via `CONVERT_FAMILY` and assert byte-equality against the committed `npp/src/convert_generated.rs`. Confirm it passes. This makes the committed artifact self-verifying and guards CUDA bumps — and lands in the same commit as the regenerated file, so every commit in the sequence is green.

Run `cargo fmt` and `cargo clippy -- -D warnings`; fix all.

### Step 2: Representative golden — 16u→32f

Create `npp/tests/golden_convert_16u32f.rs` following `golden_convert.rs` structure (`#![cfg(feature = "gpu")]`, `stream_context_for(0)`, `CudaImage::from_host(ctx, 3, W, H, &input)` for a `u16` source, `CudaImage::<f32>::new(ctx, 3, W, H)` dest, `.convert(&mut dst)`, `Vec::<f32>::try_from(&dst)`, `assert_golden`). Small deterministic `u16` gradient (12×8, 3ch). Include the "Manual procedure to pin the golden reference" doc block. Leave `EXPECTED` as a placeholder for GPU pinning — **do not fabricate values**.

### Step 3: Representative golden — 8u→16u (non-float destination)

Create `npp/tests/golden_convert_8u16u.rs`, same structure, `u8` → `u16` widening (`CudaImage::<u16>::new` dest, read back `Vec<u16>`). Exercises the dual-type `dst_step_bytes` (`size_of::<u16>`) path with a non-float dest. Deterministic `u8` gradient; document manual pinning; `EXPECTED` placeholder. Keep `golden_convert.rs` (8u→32f C3) unchanged — these three are the representative subset.

### Step 4: Verify the coverage gate

**[O2]** Run `nix develop . --command cargo tarpaulin` (non-GPU) and confirm coverage is **≥ 90%**. The generated Convert impls are expected to be FFI-dead under non-GPU tarpaulin (like the other three families) and contribute nothing to the uncovered set; this step verifies that assumption empirically. If — contrary to expectation — coverage drops, **first** investigate why Convert differs from the existing families before reaching for an annotation; only add `#[cfg(not(tarpaulin_include))]` if the existing families turn out to rely on it too (they do not, per Phase 3 Step 1). This step is the gate's checkpoint.

---

## Phase 5: Documentation reconciliation

Commit message: docs: reconcile F5.1 ConvertTo codegen across docs

### Step 1: Update codegen-architecture.md for the dual-type Convert family

In `docs/codegen-architecture.md`: add a "Convert" row to "Families implemented" (Prefix `nppiConvert_`, Channels C1/C3/C4, Custom variants —, Buffer prefix —, Shape `SRC+STEP, DST+STEP, SIZE`). Add a "Dual-type families" subsection: Convert carries two type tokens; `dual_type: true` switches the generator to `classify_convert` and the two-type emit (`impl_convert_for!($src_ty, $dst_ty, "$src_tok", "$dst_tok", { … })`); the two-token split rule (exactly-one-valid-split); `16f` excluded. Update the `FamilyDescriptor` code block to include `dual_type`. Extend "How to add a new family" with a note that dual-type families set `dual_type` and provide a two-type macro.

### Step 2: Create a first-class F5.2 roadmap entry and mark the F5.1 ConvertTo slice complete

**[F5.2]** In `docs/roadmap.md`:
- Update the **F5.1** section: the **ConvertTo** generalization across the full NPP-provided no-rounding `(src,dst)` alphabet is complete (fixture-driven via `nppiConvert_symbols.txt` → `CONVERT_FAMILY` → `convert_generated.rs`). Add a "Committed artifacts added by F5.1" bullet list (`npp-codegen/tests/fixtures/nppiConvert_symbols.txt`, `npp-codegen/examples/gen_convert_impls.rs`, `npp/src/convert_macros.rs`, `npp/src/convert_generated.rs`, `npp/tests/golden_convert_16u32f.rs`, `npp/tests/golden_convert_8u16u.rs`), and note the hand-written `ConvertTo<f32> for CudaImage<u8>` was removed from `convert_ops.rs` in favour of the generated impl while `Normalize` stayed hand-written. Point forward to F5.2.
- Add a dedicated **`## F5.2`** section (a real catalog entry alongside F5/F5.1). **What:** (1) generalize `Normalize` across the alphabet — pulls in the `nppiMulC_` family, per-source-type scale constants (`1/255`, `1/65535`, …), and float-source handling; (2) add the rounding-mode `nppiConvert_*` variants (the 20 `NppRoundMode` + 17 scaled functions), which requires a **public API change** to the `ConvertTo` trait (a rounding parameter). **Why deferred:** F5.1 chose the no-rounding `ConvertTo` group to stay shippable without an API change; each piece is ~the size of F5.1. **Dependencies:** F5.1 (establishes the dual-type codegen path), F1/F2. Update the "Suggested rough sequencing" tree and legend so F5.2 appears.

---

## Risks & callouts

1. **Fixture accuracy & build-determined coverage** — the load-bearing risk. Guarded *twice*: the Phase 4 Step 1 build (unresolved-path errors on bad symbols) and the `convert_syn_shape_check` test. The fixture is explicitly a **draft** until both pass; if a `(src,dst)` pair turns out not to fit the no-rounding shape (e.g. it secretly needs a scale arg), the resolution is to **drop it from the fixture** — self-correcting, but it means F5.1's final coverage is knowable only after a green CUDA-host build. There is no separate rollback needed: shrinking the fixture *is* the abort path. I could not enumerate symbols from source (`bindings.rs` is gitignored in `OUT_DIR`), so confirmation requires that build.

2. **`_Ctx` pivot regression [B2]** — the most dangerous subtle bug: storing the bare variant would silently drop Convert off the F8 stream path. Closed by the Phase 1 Step 1 variant-storage rule + the Phase 2 Step 4 `_Ctx`-path assertion + the prefer-`_Ctx` unit test.

3. **Two-token ambiguity [G1]** — closed by the exactly-one-valid-split algorithm + build-time panic on multiplicity + the split-robustness test.

4. **Compile breakage from the new field [B1]** — closed by naming both `ClassifiedSymbol` construction sites (incl. `classify.rs:123`) and using `Option<String>`.

5. **Byte-identity drift [G2]** — closed by adding `convert_generated_is_byte_identical` in Phase 4 (matching project convention, introduced only after regeneration so every commit is green).

6. **Coverage gate [O2]** — explicitly checked in Phase 4 Step 4 with a concrete remediation path; the exclusion mechanism is resolved against how existing generated families handle it rather than assumed (ratified: no annotation, match existing macros).

7. **`Normalize` coupling to generated `ConvertTo` [G3]** — behaviourally identical; verified by the Phase 3/4 builds and the unchanged `golden_normalize.rs`; golden left untouched by explicit instruction.

8. **No API change** — `ConvertTo` trait untouched; purely additive impls. (The API change lives in F5.2.)
