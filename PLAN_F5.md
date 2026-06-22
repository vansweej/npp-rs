# Plan: F5 — Pixel-format conversion ops (hand-written `u8 → f32` slice) — **v-final**

## Goal

Add the crate's **first cross-type operations**: `CudaImage<u8> → CudaImage<f32>` convert (`nppiConvert_8u32f_*`) and a normalize variant scaling `[0,255] → [0.0,1.0]` (convert then in-place elementwise `nppiMulC_32f_*`). Hand-written for the single `u8 → f32` pair, C1+C3 only. No codegen generalization (deferred to F5.1). Zero `npp-sys` changes. Plus: tighten the crate-wide C4 wording.

## Locked decisions
- **R2** → Normalize = Convert then `nppiMulC_32f`.
- **R3** → Generic `trait ConvertTo<Dst: NppPixelType>`, concrete `impl … for CudaImage<u8>`.
- **R4** → C1 + C3.
- **Finding 2** → **Option A**: in-place elementwise `MulC` on `dst`, documented carve-out.
- **C4 wording** → tighten crate-wide (Phase 0).

## Confirmed facts
- ✅ No `npp-sys` change — `nppi.*` wildcard covers `nppiConvert_*` + `nppiMulC_*`.
- ✅ Link deps present — `build.rs` links `nppial` (`MulC`) + convert-family libs.
- ✅ `height_stride` is element-count (`layout.rs:37`) → **dst byte-step uses `size_of::<f32>()`**.
- ✅ `assert_golden::<f32>` works (generic `T: PartialEq + Debug`).
- ✅ No crate-root trait re-export — tests import `use npp_rs::imageops::{…}`.
- ✅ C4 wording sites enumerated (4 code doc-comments + 1 docs file).
- ✅ **`convert_ops.rs` matches NO tarpaulin exclude glob** (`tarpaulin.toml` excludes `*_generated.rs`/`*_macros.rs`/etc., not `convert_ops.rs`) → coverage governed **only** by per-fn `#[cfg(not(tarpaulin_include))]`. *(Finding 4)*
- ✅ `#![deny(missing_docs)]` is active → every new trait **and method** needs `///` docs. *(Observation 7)*

---

## Phase 0: Tighten C4 wording crate-wide *(docs-only, isolated commit, do first)*

Commit message: `docs: clarify C4 invariant — gather-overlap vs elementwise aliasing`

**Rationale:** The blanket "src/dst must be non-overlapping" wording over-states C4 (the real hazard is *neighbourhood-gather* overlap). Stating it precisely **before** F5 prevents `normalize`'s in-place `MulC` from looking like a rule violation.

**Edits (enumerated sites):**
1. `npp/src/imageops.rs:28–29` (`Resize` `# Precondition`) → *"`src` and `dst` must not overlap in memory. This applies to **neighbourhood-gather** operations (e.g. resize samples a pixel window); aliasing produces undefined results. Purely **elementwise** operations may safely alias (see `Normalize`)."*
2. `npp/src/imageops.rs:46` (`SwapChannels` `# Precondition`) → same, adapted.
3. `npp/src/resize_macros.rs:23` and `:47–48` → align macro-emitted doc text.
4. `npp/src/swap_channels_macros.rs:23` and `:47–48` → same.
5. **`docs/npp-bindings.md:85–86` — the PRIMARY site** *(Finding 5)*: this is the canonical "Safety invariants" section in the "how to add an op" guide; it's more authoritative than per-trait comments. Reword to the gather-vs-elementwise distinction with the most care.

**Constraint:** pure documentation, no behavioural change. Verify: `cargo doc --no-deps -p npp-rs` (deny-missing-docs still passes) + `cargo fmt --check`.

---

## Phase 1: Convert trait + `u8 → f32` impl

Commit message: `feat: add cross-type ConvertTo trait and u8→f32 convert`

**Step 0 (R1 verification gate — first executable action):** Grep generated `bindings.rs` in `OUT_DIR`; confirm before any call site: `nppiConvert_8u32f_C1R_Ctx` / `_C3R_Ctx`, `nppiMulC_32f_C1R_Ctx` / `_C3R_Ctx`, **and `MulC` constant arity** (C1 scalar `nConstant: f32` vs C3 `aConstants: *const f32` len-3).

**Step 1:** In `imageops.rs`, add `pub trait ConvertTo<Dst: NppPixelType> { fn convert(&self, dst: &mut CudaImage<Dst>) -> Result<(), NppError>; }`. **Doc the trait AND the `convert` method** *(Observation 7)*: first cross-type op; non-overlap (gather wording from Phase 0); dst step uses dst element type; `InvalidArgument` on mismatch, `Npp` on failure.

**Step 2:** New `npp/src/convert_ops.rs`, `impl ConvertTo<f32> for CudaImage<u8>`:
- **Two distinct checks**: (a) agreement — `channels/width/height` all equal else `InvalidArgument("…dimensions/channels must match…")`; (b) support — channels ∈ {1,3} via `match` `_` arm with its own message.
- `NppiSize` from dst; `src_step = height_stride * size_of::<u8>()`; **`dst_step = height_stride * size_of::<f32>()`**; pointer bridge offset by each `img_index`.
- `match self.channels()`: `1 => nppiConvert_8u32f_C1R_Ctx`, `3 => nppiConvert_8u32f_C3R_Ctx`; trailing `self.ctx.raw_ctx()`; `check_status`.
- **`#[cfg(not(tarpaulin_include))]` on the impl method.**

**Step 3:** `lib.rs` — add `pub mod convert_ops;` (doc-commented). **No `pub use`.** Mention `ConvertTo` in crate docs.

**Step 4:** `npp/tests/golden_convert.rs` (GPU-gated, `use npp_rs::imageops::ConvertTo;`): C3 `u8` gradient → C3 `f32` → `convert` → `Vec<f32>` → `assert_golden::<f32>(…, "convert_8u32f")`, `EXPECTED: &[f32] = &[]` (ship red). FP-exact.

**Step 5 (coverage gate — Finding 4):** Confirm **every** function in `convert_ops.rs` carries `#[cfg(not(tarpaulin_include))]`; run `cargo tarpaulin` and verify the ≥ 90 % gate still holds.

---

## Phase 2: Normalize via convert + in-place MulC

Commit message: `feat: add u8→f32 normalize via convert and in-place MulC scale`

**Step 1:** `imageops.rs` — `pub trait Normalize<Dst: NppPixelType> { fn normalize(&self, dst: &mut CudaImage<Dst>) -> Result<(), NppError>; }`. **Doc trait AND method**: `[0,255] → [0,1]` for NN preprocessing; convert-then-scale; convert step obeys non-overlap, scale step elementwise-in-place (cross-ref Phase 0 wording).

**Step 2:** `convert_ops.rs` — `impl Normalize<f32> for CudaImage<u8>`:
1. `self.convert(dst)?` (writes `[0,255.0]`).
2. In-place `nppiMulC_32f_*_Ctx` on `dst` by `1/255`, **channel-correct arity**: C1 scalar `1.0/255.0`; **C3 `aConstants: [f32;3] = [1.0/255.0; 3]`** per confirmed signature. dst step = `height_stride * size_of::<f32>()`; `match dst.channels()`; trailing `dst.ctx.raw_ctx()`; `check_status`.
3. **Carve-out comment at call site:** *"`MulC` is elementwise — output (x,y) depends only on input (x,y) — so reading/writing `dst` in place is sound and does not trigger the C4 gather-overlap hazard."*
4. **`#[cfg(not(tarpaulin_include))]`.**

**Step 3:** `npp/tests/golden_normalize.rs` (GPU-gated, `use npp_rs::imageops::Normalize;`): C3 `u8` with `0/128/255` → `normalize` → `Vec<f32>` → assert pinned exact `f32` (`0.0`, `0.50196…`, `1.0`), `EXPECTED: &[f32] = &[]` (ship red). Deterministic → bit-exact.

**Step 4 (coverage gate):** Re-run `cargo tarpaulin`; confirm ≥ 90 % after the second impl lands.

---

## Phase 3: Documentation

Commit message: `docs: document F5 cross-type convert/normalize slice`

- `docs/roadmap.md`: F5 `*(slice complete)*`; carve **F5.1 — cross-type Convert codegen**.
- `docs/architecture.md`: "cross-type operations" note + `convert_ops.rs` row in module table.
- **`docs/npp-bindings.md`** *(Finding 5)*: add `ConvertTo`/`Normalize` to the op coverage area — but **marked hand-written, NOT in the "generated impls" table** (lines 30–32 are explicitly codegen output). Add a separate "Hand-written ops" sub-table or clearly annotate the rows as `convert_ops.rs` (hand-written), so the doc doesn't imply codegen.

---

## Testing posture (explicit, per Observation 6)

`npp-bindings.md` step 6 names a "geometry assertion test" as the minimum for a new op. F5's `convert_ops.rs` impls are **fully GPU/FFI** (every fn `#[cfg(not(tarpaulin_include))]`), and the validation checks (`agreement`/`support`) require a constructed `CudaImage`, which requires a device — so **the dimension-mismatch → `InvalidArgument` path is exercisable only on-GPU**. Decision: the two **golden tests satisfy step 6's intent** (they assert correctness, a superset of geometry); we deliberately add **no separate non-GPU unit test**, because the validation logic isn't reachable without a device. This is a **known, named gap consistent with the rest of the crate** (Resize/SwapChannels/Mean validation is likewise GPU-only-tested). *If* you later want the validation path unit-testable off-GPU, that requires refactoring the checks to operate on `CudaLayout` alone — out of scope for this slice; flag for F6 if desired.

## Verification (Nix shell, per phase)
`cargo build` · `cargo test` (unit, no GPU) · `cargo clippy -- -D warnings` · `cargo fmt --check` · `cargo doc --no-deps -p npp-rs` · `cargo tarpaulin` (≥ 90 %, gated after Phases 1 & 2). GPU goldens = manual pin gate, shipped red.

## Residual risk
**R1** — symbol/signature/arity confirmation, resolved by Phase 1 Step 0. Low, surfaced.

## Strategy summary
Five-commit feature. **Phase 0:** tighten the C4 invariant crate-wide (5 sites, `npp-bindings.md:85` is primary; docs-only, isolated) so in-place elementwise ops are principled. **Phase 1:** hand-written `u8→f32` `ConvertTo`, with explicit per-fn tarpaulin annotation + coverage gate. **Phase 2:** `Normalize` as convert + in-place `MulC(1/255)`, channel-correct constant arity, documented elementwise carve-out, coverage re-gate. **Phase 3:** docs + F5.1 deferral, with `ConvertTo`/`Normalize` marked **hand-written** (not codegen) in `npp-bindings.md`. Testing posture: goldens satisfy the geometry-test minimum; validation path is GPU-only by construction (named gap). Net surface: clarified C4 docs, two trait+method-documented additions, one `convert_ops.rs`, two goldens — **zero `npp-sys` changes, no re-exports, distinct checks, Option-A aliasing, coverage-gated, deny-missing-docs satisfied.**
