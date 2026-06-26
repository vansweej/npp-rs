# npp-rs Roadmap

> Work after Milestone 1 ("build again on latest CUDA via Nix + cudarc").
> Each feature is a candidate for its own sparring and planning session.
> Prioritization and sequencing will be decided per session; this doc is
> a descriptive catalog, not a commitment schedule.

Milestone 1 is **complete and merged** — cudarc 0.9 port, Nix dev shell, shared NPP linking,
bindgen `nppi*` FFI, and GPU-gated tests. This roadmap is now the single forward source of
truth for all post-M1 work.

---

## Resolved decisions (binding — do not re-litigate)

These decisions are inherited by F1/F2 (notably `nppi*`-only / `npps*`-deferred),
so they must live in the roadmap before the M1 plan document is deleted.

| Decision | Resolution |
|----------|-----------|
| CUDA major | Latest in nixpkgs — `nixos-unstable`, `pkgs.cudaPackages` |
| CUDA crate | `cudarc 0.9`, `default-features = false`, features `["driver","std"]` — replaces `rustacuda*` |
| GPU tests | Manual only — feature-gated behind `gpu`; plain `cargo test` skips them |
| NPP linking | Shared (dynamic) — no `static=` names |
| Platform | Linux only — no Windows paths |
| NPP nixpkgs attr | `cudaPackages.libnpp` |
| Binding philosophy | Safe, idiomatic Rust; rewrite rather than faithfully port; current files are behavioural reference |
| Domain scope | NPP image ops (`nppi*`) only; signal ops (`npps*`) deferred to F9 |

## Out of scope for M1 (deferred)

- C2 — replace `debug_assert!` with `Result`-returning validation and seal the format
- C5 — replace `Vec::with_capacity` + `set_len` with zeroed/`MaybeUninit` + stride fix
- ~~C8 — stream/execution-context model (CUDA streams, async ops)~~ **→ Delivered by F8 (core)**
- C11 — (if deferred by the open decision) seal/remove generic `T`
- C12 — golden-image correctness tests
- IPP bindings (`ipp-sys`/`ipp`)
- NPP signal ops (`npps*`)
- `image` crate upgrade from `0.23.13` to modern
- Broadening NPP coverage or adding pixel formats

---

## F1 — Macro-generated binding codegen *(implemented)*

**What:** Replaced the hand-written `u8`/`f32` capability-trait impls from M1
with macro-generated impls covering the full `NppPixelType` alphabet that has
NPP resize symbols.

**Architecture — two derivation pipelines:**

The codegen uses two independent pipelines for the two axes of capability:

| Axis | What | Source | Policy |
|------|------|--------|--------|
| `type × channel` | Which NPP symbols exist for a given `(T, C)` | Scraped from committed corpus (`tests/fixtures/nppiResize_symbols.txt`) → `suffix_classifier::classify` → `examples/gen_resize_impls`. Committed output: `src/resize_generated.rs` | Re-run generator on CUDA bump |
| `type × interpolation` | Which interpolation modes are supported for a given `(T, mode)` | Probed at runtime via `tests/probe_resize_caps.rs` (GPU-gated). Committed output: `src/resize_caps.rs` | Re-probe on CUDA bump |

**Key structural decisions:**
- **Channel count is runtime data**, not a type parameter (`CudaImage::new(device, channels: u8, …)`
  stored in `layout.channels`). The macro dispatches via `match self.channels()` to C1/C3/C4 symbols.
- **Mode safety is runtime-checked** against the committed `RESIZE_CAPS` table via `mode_supported()`,
  not a compile-time error.
- **`16f` is excluded** from the safe layer (`half` crate disabled). The alias probe exercises it via
  raw `npp_sys` FFI but the generator skips it.
- **The status-code spike** (`tests/spike_npp_status.rs`) is a committed CUDA-bump regression guard,
  not a throwaway diagnostic. Three pinned NppStatus codes: success=0, interpolation-error=-22,
  harness-bug=-201.

**New committed artifacts:**
- `src/suffix_classifier.rs` — pure string-parsing `classify()` function with 18 offline tests.
- `src/resize_macros.rs` — `impl_resize_for!` macro (crate-wide via `#[macro_export]`).
- `src/resize_generated.rs` — macro invocations for `u8`, `u16`, `i16`, `f32`, each with C1/C3/C4 arms.
- `src/resize_caps.rs` — GPU-probed `(type, interpolation)` support matrix.
- `tests/fixtures/nppiResize_symbols.txt` — captured corpus from CUDA 12.9.
- `tests/spike_npp_status.rs` — status-code taxonomy pinning (GPU-gated).
- `tests/probe_resize_caps.rs` — the GPU probe harness (GPU-gated).
- `examples/gen_resize_impls.rs` — generator that reads the corpus and emits `resize_generated.rs`.

**Generated (gitignored):** `npp-sys/src/bindings.rs` (bindgen, every build).

**Dependencies:** M1 (needed the proven trait exemplars and the validated FFI pointer bridge).

---

## F2 — Expand `NppPixelType` operation coverage to the rest of the alphabet *(complete)*

**What:** Once F1's macro works, fill in `Resize`, `SwapChannels`, and `Mean`
capability traits for the full `NppPixelType` alphabet wherever NPP provides
the symbol.

**Why:** M1 ships the *alphabet* (all types constructible) but only `u8`/`f32`
*ops*. F2 closes this gap — driven by the macro codegen crate (`npp-codegen`).

**Architecture — dedicated `npp-codegen` crate:**

The F1 suffix classifier was split out into a standalone `npp-codegen/` crate
with three core modules (`classify`, `shape`, `gen_impls`) and a `survey_shapes`
binary. The classifier was generalized with two key additions needed for the
SwapChannels and Mean families:

| Addition | Purpose | Used by |
|----------|---------|---------|
| `custom_variants` | Accept non‑standard channel suffixes (e.g. `C4C3R`) | SwapChannels |
| `get_buffer_host_size_prefix` | Emit `(mean_sym, buffer_sym)` tuples for two‑call scratch‑buffer ops | Mean |

See `docs/codegen-architecture.md` for full details on `FamilyDescriptor` and
the generation flow.

**Phases (implemented on `feat/f2-codegen-phase1`):**

| Phase | What | Commit |
|-------|------|--------|
| 1 | npp-codegen crate + shape survey | `710df45` |
| 2 | Generalized classifier + generator, delete `suffix_classifier.rs` from npp-rs | `cf3eced` |
| 3 | Golden‑test helper (`test_helpers::assert_golden`) | `434b6af` |
| 4 | SwapChannels macro generation (C4C3R 4→3 conversion) | `cb49c21` |
| 5 | Mean reduction (two‑call scratch‑buffer dance) | `5763e4f` |
| 6 | Documentation reconciliation | `da6d156` |

**Committed artifacts added by F2:**
- `npp-codegen/` — entire crate (classify, shape, gen_impls, survey_shapes binary)
- `npp-codegen/tests/fixtures/nppiSwapChannels_symbols.txt` — C4C3R fixture
- `npp-codegen/tests/fixtures/nppiMean_symbols.txt` — C1/C3/C4 fixture
- `npp-codegen/examples/gen_swap_channels_impls.rs`
- `npp-codegen/examples/gen_mean_impls.rs`
- `npp/src/swap_channels_macros.rs`
- `npp/src/swap_channels_generated.rs`
- `npp/src/mean_macros.rs`
- `npp/src/mean_generated.rs`
- `npp/tests/golden_swap_channels.rs`
- `npp/tests/golden_mean.rs`

**Deleted:** `npp/src/suffix_classifier.rs`, `npp/src/swap_channel_ops.rs`

**Open items (deferred):**
- The `Resize` family still uses `16f`-skip and has no golden test for non‑u8 types.
- Golden tests for all three families (Resize, SwapChannels, Mean) are pinned and GPU-verified.
- `32u`, `64f`, `8s` have no `nppiMean_*` symbols with standard channel variants.
- `npp-codegen` does not handle the `Mean_StdDev` or other compound op families yet.

**Dependencies:** F1.

---

## F3 — `image-rs` boundary integration *(dropped from this repository)*

**What:** Re-introduce `image` crate conversions as *boundary* `From`/`TryFrom`
edges (not woven into the core): `TryFrom<&RgbImage> for CudaImage<u8>`,
`TryFrom<&CudaImage<u8>> for RgbImage`, and the `f32` equivalents
(`ImageBuffer<Rgb<f32>>`).

**Why:** M1 strips `image` out of the core entirely (round-trip is `Vec<T>`
only). This adds the convenience layer back at the edges, preserving the design
constraint that image types stay at the interop boundary.

**Carries a real sub-decision:** target is modern `image` (0.25+), whose
`Pixel`/`Primitive` API differs from the old 0.23. Whether `NppPixelType`
should *derive from* / map onto image-rs's `Primitive` trait family, or stay
independent with explicit conversions, is an open design question to spar on —
not something to design against 0.23.

**Dependencies:** M1 (core must be `image`-free first). Independent of F1/F2
(can wrap just `u8`/`f32` boundaries early).

**Status change (F6.2):** Ecosystem integration for the `image` crate moves to
a **separate downstream integration repository** that consumes `npp-rs` as a
published crate. Any public API that repository requires from `npp-rs` will be
decided as a deliberate future feature when it materialises.

---

## F4 — `graphynx` interop *(dropped from this repository)*

**What:** `From`/`Into` boundary conversions between `CudaImage<T>` and
graphynx's own device-buffer/element types (the author's cudarc-based compute
library).

**Why:** One of the three stated use cases. Because both `npp-rs` and graphynx
are built on cudarc `CudaSlice`, this should be a relatively thin boundary
(same underlying primitives) — but it's an integration with its own
type-matching work.

**Open question:** whether the conversion is zero-copy (share the underlying
`CudaSlice`/`CUdeviceptr`) or copying, and what ownership/lifetime contract
crosses the boundary.

**Dependencies:** M1. Independent of F1–F3.

**Status change (F6.2):** Ecosystem integration for the `graphynx` crate moves
to a **separate downstream integration repository** that consumes `npp-rs` as a
published crate. Any public API that repository requires from `npp-rs` will be
decided as a deliberate future feature when it materialises.

---

## F5 — Pixel-format conversion ops (`u8 ↔ f32`, scaling) *(slice complete)*

**What:** On-device pixel *conversion* operations — e.g.
`RgbImage<u8> → RgbImage<f32>`, normalization/scaling — backed by NPP's
`nppiConvert_*` / `nppiMulC_*` families.

**Status:** Hand-written `u8 → f32` slice complete (commits `9cb8529`–`cbac264`).
Committed artifacts: `npp/src/convert_ops.rs`, `npp/src/imageops.rs` (ConvertTo/Normalize traits),
`npp/tests/golden_convert.rs`, `npp/tests/golden_normalize.rs`. Supports C1 and C3 channels.
Normalize implements [0,255] → [0.0,1.0] scaling for NN preprocessing via convert + in-place MulC.

**Deferred to F5.1:** Codegen generalization to cover the full `NppPixelType` alphabet
and all supported channel counts (C4, etc.). Hand-written slice establishes the pattern
and validates the two-trait design (ConvertTo + Normalize).

**Dependencies:** F1/F2 (macro infrastructure), but hand-written slice was prioritized
for the NN-preprocessing use case.

---

## F5.1 — Cross-type ConvertTo codegen *(complete)*

**What:** Generalized the hand-written `u8 → f32` `ConvertTo` implementation to
**every `(src, dst)` pixel-type pair that NPP provides a non-rounding `nppiConvert_*_Ctx`
symbol for**, using the dual-type codegen pattern (fixture → `classify_convert` →
`CONVERT_FAMILY` → `convert_generated.rs`). The hand-written `Normalize<f32> for
CudaImage<u8>` stays. No public API change.

**Why:** F5's hand-written slice validates the trait design and delivers the immediate
NN-preprocessing need. F5.1 extends it to all types and channels, following the proven
codegen pattern — with the new dual-type mechanism (Convert carries two type tokens).

**Dependencies:** F5 (establishes the trait and pattern), F1/F2 (macro infrastructure).

**Committed artifacts added by F5.1:**
- `npp-codegen/src/classify.rs` — `classify_convert()` dual-type classifier
- `npp-codegen/tests/fixtures/nppiConvert_symbols.txt` — no-rounding Convert fixture
- `npp-codegen/examples/gen_convert_impls.rs` — generator example
- `npp/src/convert_macros.rs` — `impl_convert_for!` dual-type macro
- `npp/src/convert_generated.rs` — 7 `(src,dst)` groups: `(i16→f32, u16→f32, u16→i32, u8→i16, u8→u16, u8→f32, u8→i32)`, all `_Ctx`
- `npp/tests/golden_convert_16u32f.rs` — golden test for 16u→32f (pinned)
- `npp/tests/golden_convert_8u16u.rs` — golden test for 8u→16u (pinned)

The hand-written `ConvertTo<f32> for CudaImage<u8>` was removed from `convert_ops.rs`
in favour of the generated impl. `Normalize` stayed hand-written (generalization
deferred to F5.2).

**Generated (from fixture, committed):** `npp/src/convert_generated.rs` — byte-identity
guarded by `convert_generated_is_byte_identical` test.

**Point forward to F5.2:** Normalize generalization across the alphabet and
rounding-mode Convert variants are the next step.

---

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

---

## F5.3 — Rounding-mode `ConvertRounded<Dst>` (additive) *(implemented)*

**What:** Add the rounding-mode `nppiConvert_*` variants — the narrowing
conversions with explicit `NppRoundMode` parameter (shape `SRC+STEP, DST+STEP,
SIZE, MISC:NppRoundMode`) — as a **new, non-breaking `ConvertRounded<Dst>`
capability trait**, not a modification to `ConvertTo`.

**Key design decisions:**
- **Additive**, not breaking — a separate `ConvertRounded<Dst>` trait, matching
  the one-trait-per-capability house style (`ConvertTo`, `Normalize`).
- **Round-mode group only**; scaled variants (`+ CONST_SCALAR`) deferred to F5.4.
- **Separate fixture and classifier** (`nppiConvertRound_symbols.txt` +
  `classify_convert_round`) because round-mode symbols share `{src}{dst}_CxR`
  names with no-rounding Convert symbols, and `classify_convert` is already used
  by both `CONVERT_FAMILY` and Normalize against the same fixture.
- **`dual_type_round` selector** on `FamilyDescriptor` — when `true`,
  `generate_for_family()` dispatches to `classify_convert_round` instead of
  `classify_convert`.
- **The macro injects the round-mode argument** — `impl_convert_rounded_for!`
  inserts `round_mode(mode)` between the `NppiSize` param and the stream context.

**Committed artifacts:**
- `RoundMode` enum, `ConvertRounded<Dst>` trait in `imageops.rs`
- `convert_round_ops.rs` translator, `convert_round_macros.rs` macro
- `convert_round_generated.rs` (4 narrowing pairs: `f32→{u8,i8,u16,i16}`)
- `classify_convert_round` in `classify.rs`, `CONVERT_ROUND_FAMILY` in `gen_impls.rs`
- `nppiConvertRound_symbols.txt` fixture, `gen_convert_round_impls.rs` example
- GPU-gated golden tests (3 per-mode + 1 chained `_Ctx`)

**Dependencies:** F5.1 (dual-type codegen path) and F5.2 (Normalize precedent).

---

## F5.4 — Scaled rounding-mode `nppiConvert` variants *(complete)*

**What:** Bind the `nppiConvert_*C1RSfs` family — single-channel pixel-type
conversion with a rounding mode and a scale factor. Ships a
`ConvertRoundedScaled<Dst>` trait following the F5.3 codegen pattern.

**Scope note:** Only single-channel (`C1RSfs`) — NPP does not expose
`C3RSfs`/`C4RSfs`.

**D-API note:** The `nScaleFactor` parameter is documented as `"Integer Result
Scaling"` in the NPP docs. Per NPP `Sfs` convention, this is presumed to be a
power-of-two exponent (`2^nScaleFactor`).

**Committed artifacts added by F5.4:**
- `npp-codegen/src/classify.rs` — `classify_convert_round_scaled` + unit tests
- `npp-codegen/src/gen_impls.rs` — `CONVERT_ROUND_SCALED_FAMILY` descriptor,
  `dual_type_round_scaled` dispatch, generator tests (byte-identity + corpus)
- `npp-codegen/examples/gen_convert_round_scaled_impls.rs` — generator example
- `npp-codegen/tests/fixtures/nppiConvertRoundScaled_symbols.txt` — fixture
- `npp/src/convert_round_scaled_macros.rs` — `impl_convert_rounded_scaled_for!`
- `npp/src/convert_round_scaled_generated.rs` — 17 generated invocations
- `npp/src/imageops.rs` — `ConvertRoundedScaled<Dst>` trait
- `npp/tests/golden_convert_round_scaled.rs` — shape-check test
- `npp/tests/golden_convert_round_scaled_pinned.rs` — per-mode golden test (3 modes)
- `npp/tests/golden_convert_round_scaled_chained.rs` — chained `_Ctx` golden test
- Phase 0 bug fixes: unified `find_bindings` resolver, `example_name` field on
  `FamilyDescriptor`

**Dependencies:** F5.3 (`CONVERT_ROUND_FAMILY` infrastructure, `RoundMode` enum).

---

## F6 — Correctness hardening & doc-hygiene *(complete)*

**What:** Fix a latent `img_index` pointer-arithmetic bug in all five op
macros, fill the named Resize‑non‑u8 golden gap (i16, u16, f32), add the
CT5 non‑aligned‑width stride‑hazard test, add one correctness‑asserting
Resize benchmark, add the two missing byte‑identity guards (mean,
swap_channels), and reconcile the doc claims that had drifted false (C12
statement, stale "stub" reference, F6 scope exaggeration).

**Key design note:** The accessor abstraction (`NppImageRef`/`NppImageMut`),
`*_into` engines, and ROI golden tests were designed in the F6 planning
session but **deferred to F6.2** to keep F6 focused on the concrete,
independently‑releasable hardening work.

**Committed artifacts added by F6:**
- Bug fix: five `*_macros.rs` files (remove `+ img_index` addend)
- Guards: `mean_generated_is_byte_identical`, `swap_channels_generated_is_byte_identical`
  tests in `npp-codegen/src/gen_impls.rs`
- Goldens: `npp/tests/golden_resize_16s.rs`, `golden_resize_16u.rs`,
  `golden_resize_32f.rs`
- CT5: `npp/tests/ct5_non_aligned_width.rs`
- Bench: `npp/benches/bench_resize_correctness.rs`
- Doc fixes: AGENTS.md C12 line, roadmap stale claims, F6.2 entry

---

## F6.2 — ROI sub-image support for Resize and SwapChannels *(implemented)*

**What:** `pub(crate)` macro‑emitted `*_into` engine functions for Resize and
SwapChannels, view readback helper (`TryFrom<&CudaImageView<T>> for Vec<T>`),
and in‑crate ROI golden tests exercising the engines via
`CudaImageView::device_ptr()`. Designed during the F6 planning session.

**Scope:** Resize and SwapChannels only. Convert/Normalize/Mean remain
owned-buffer only. No public API added — engines are `pub(crate)`, the view
readback is behind the `CudaImageView` type.

**Dependencies:** The `img_index` fix from F6 (pointer maths correct on owned
path). cudarc 0.9.15 `dtoh_sync_copy_into` confirmed to accept `&CudaView` via
`Src: DevicePtr<T>`.

---

## F6.1 — Benchmark port

The five `npp/benches/*.rs` files from the original crate use `rustacuda`,
`image-rs`, and `cuda-runtime-sys` — none of which exist in the M1 dependency
set. They are parked (not built) and must be reimplemented as full benchmarks
asserting both timing and output content. Depends on M1's new API.

---

## F7 — Release-mode validation hardening (C2)

**What:** Replace `debug_assert!` format/stride preconditions at the FFI
boundary with `Result`-returning validation (or `assert!`), so safety checks
survive `--release`.

**Why:** C2 was the report's one genuinely-reachable release-mode memory
hazard. M1's type seal (C11) **already removes the worst path** — you can no
longer construct a wrong-channel `CudaImage` via an open `ColorType`, because
the type *is* the format. What remains: residual stride/dimension invariants
not expressible purely in the type (e.g. src/dst dimension agreement for
`bgra_to_rgb`, ROI extent), and stride-alignment enforcement at construction
(the CT5 fix — carry a real pitch and reject mis-aligned widths). Lower urgency
post-M1 than pre-M1, but not gone.

**Dependencies:** M1 (the type seal does half the job; this finishes it).

---

## F8 — Stream / execution-context model (C8 + C7) *(core complete)*

**What:** A first-class stream/execution-context abstraction: `StreamContext`
(forked stream + populated `NppStreamContext`), the `_Ctx` pivot of Resize/SwapChannels/Mean,
the host-fenced NULL-stream readback contract, C7 tie via the `ctx` field on
`CudaImage`, and `!Send + !Sync` enforcement.

**Status:** Core implementation merged on `main` (commits `a7456c7`–`f6b1d5c`).
Committed artifacts: `npp/src/stream.rs`, `npp-sys/tests/stream_context_symbols.rs`,
`docs/stream-context.md`. All three ops (Resize, SwapChannels, Mean) pivot to
`_Ctx` variants via macro regeneration (commit `257bda6`). Golden tests re-pinned
and passing.

**Cross-cutting flag — RESOLVED:** F8 was flagged as interacting with F1 (macro).
The feared risk was that F1's macro templates would ossify before F8's signature
shape was decided, forcing a regeneration when streams landed. **This event already
occurred and is closed:** F8 shipped *after* F1/F2 were complete, and the macros
were regenerated onto `_Ctx` symbols in a single coordinated pass (commit `257bda6`
"accept and prefer `_Ctx` variants"). The sequencing constraint is satisfied.

**Two sub-goals deferred to F8.1 and F8.2** (see below).

**Dependencies:** M1.

---

## F8.1 — Configurable device selection *(complete)*

**What:** Configurable device selection was already delivered by F8 core — `CudaImage`
is constructed from an `Arc<StreamContext>`, and `stream_context_for(ordinal)` already
requires an explicit ordinal. F8.1's actual change is the removal of the unused
`default_cuda_device()` convenience wrapper function that was never re-exported and
never referenced in consumer documentation.

**Why:** F8's original scope included removing a hardcoded device-0 path. The core
stream abstraction (the `StreamContext` pivot) already satisfied that requirement;
`default_cuda_device()` was an unused vestige. This is dead-code cleanup.

**Dependencies:** F8 (core).

---

## F8.2 — Compute/copy overlap / async multi-stream chaining

**What:** Enable compute/copy overlap and multi-stream async chaining. The `StreamContext::device_fence()`
method already exists (calls `cuDeviceGetAttribute` via `CudaDevice::wait_for`), providing
device-side ordering without host blocking. This feature would extend the async contract
to support chaining operations across multiple streams on the same device, with explicit
device-side fences between them.

**Why:** F8's original scope included "compute/copy overlap" — the entire performance reason
to use CUDA asynchronously. The core stream abstraction (host-fenced readback, forked stream
per context) landed, but async chaining was deferred to "Session 3 (future)" per the F8
session briefs. This is the remaining work to unlock true async pipelines.

**Dependencies:** F8 (core).

---

## F9 — NPP signal ops (`npps*`)

**What:** Extend bindings to NPP's signal-processing domain (`npps*` symbols),
currently excluded (`wrapper.h` comments out `npps.h`; bindgen allowlist is
`nppi*`-only).

**Why:** Explicitly out of scope for the image milestones, but a stated
long-term direction. Would mirror the `nppi*` architecture (allowlist,
`NppPixelType`-analogous typing, capability traits, macro codegen).

**Dependencies:** Wants F1's macro infrastructure to exist first (otherwise
you're hand-wrapping a second large domain).

---

## F10 — IPP bindings *(separate project / out of scope)*

**What:** `ipp-sys` / `ipp` sibling crates for Intel IPP, paralleling the
`npp-sys`/`npp-rs` split — the CPU counterpart to the GPU NPP ops.

**Why:** Entirely separate dependency and platform story (Intel IPP, not CUDA);
shares only the *architectural pattern* (sys-crate + safe-wrapper, capability
traits, codegen), not code. This repository focuses on NVIDIA GPU operations;
Intel IPP belongs in its own repository.

**Dependencies:** Conceptually independent; benefits from F1's macro/codegen
patterns being mature enough to reuse the approach.

---

## Suggested rough sequencing

```
M1 ──┬─> F1 (macro codegen) ──> F2 (alphabet coverage) ──> F5 (convert ops) ──> F5.1 (ConvertTo codegen) ──> F5.2 (Normalize codegen) ──> F5.3 (ConvertRounded, DONE)
     │      │                                                   └────────────────> F5.4 (scaled rounding, deferred)
     │      │                              └────────────────> F6 (correctness hardening) ──> F6.2 (ROI sub-image support, DONE)
     ├─> F7 (release validation)          [type seal did half in M1]
     ├─> F8 (streams/execution context)   [DONE]
     │      ├─> F8.1 (configurable device selection)
     │      └─> F8.2 (async multi-stream chaining)
     └─> F9 (npps signal ops) ── after F1
```

**Sequencing note:** F1, F2, F5.1, F5.2, F5.3, F6, F6.2, F8 (core), and F8.1 are
complete and merged on `main`. F3 (image-rs) and F4 (graphynx) are **dropped from
this repository** — ecosystem integration moves to separate downstream repos.
F10 (IPP) is a separate project. The cross-cutting F8↔F1 signature-shaping risk
(the load-bearing constraint that shaped the original roadmap) is **resolved**:
F8 shipped after F1/F2 with a clean `_Ctx` regeneration, so the feared "regenerate
when streams land" event already occurred and is closed. All remaining features
(F5.4, F6.1, F7, F8.2, F9) are independent — the next phase is a free choice.

---

## Legend

| Label | Source | Status |
|-------|--------|--------|
| C1–C12 | Architecture review finding (`reviews/final-report.md`) | Various — C1, C2, C5, C12, C7, C8 referenced above |
| CT1–CT6 | Contested finding (same report) | CT5 stride concerns and CT1 `set_len` referenced |
| M1 | Milestone 1 ("build again") | Complete; this roadmap catalogs everything after it |
