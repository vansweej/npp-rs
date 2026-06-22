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

## F3 — `image-rs` boundary integration

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

---

## F4 — `graphynx` interop

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

## F5.1 — Cross-type convert/normalize codegen

**What:** Generalize the hand-written `u8 → f32` ConvertTo/Normalize impls to cover
the full `NppPixelType` alphabet and all supported channel counts via macro codegen
(similar to F1/F2's Resize/SwapChannels/Mean pattern).

**Why:** F5's hand-written slice validates the trait design and delivers the immediate
NN-preprocessing need. F5.1 extends it to all types and channels, following the proven
codegen pattern.

**Dependencies:** F5 (establishes the trait and pattern), F1/F2 (macro infrastructure).

---

## F6 — Full golden-image correctness test suite

**What:** Byte-exact device-output-vs-reference tests for full frames **and**
sub-image ROIs, across wrapped ops and types. Make benchmarks assert
correctness, not just timing. Closes the report finding C12 (no test verifies
pixel correctness anywhere).

**Why:** M1 ships exactly *one* golden round-trip (a smoke test that the cudarc
port computes at all). This is the real coverage. Also the only thing that
validates the packed-vs-pitched stride concern (CT5) and catches
`NPP_STEP_ERROR` at non-4-aligned widths.

**Dependencies:** Grows alongside F1/F2 (each new wrapped `(type, op)` cell
wants a golden test). Needs a GPU host (manual gate, per M1's test tiering).

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

## F8.1 — Configurable device selection

**What:** Remove the hardcoded ordinal-0 path from `cuda.rs`. The `stream_context_for(ordinal)`
function already exists and accepts a device ordinal, but `default_cuda_device()` hardcodes
ordinal 0. Eliminate the hardcoded path so all device selection is explicit.

**Why:** F8's original scope included "configurable device selection (kill the hardcoded
`Device::get_device(0)`)". The core stream abstraction landed, but this sub-goal was
deferred. It is a straightforward cleanup: remove `default_cuda_device()` or make it
a thin wrapper that requires an explicit ordinal argument.

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

## F10 — IPP bindings as sibling crates *(furthest out)*

**What:** `ipp-sys` / `ipp` sibling crates for Intel IPP, paralleling the
`npp-sys`/`npp-rs` split — the CPU counterpart to the GPU NPP ops.

**Why:** Stated future vision ("IPP bindings as siblings later"). Entirely
separate dependency and platform story (Intel IPP, not CUDA); shares only the
*architectural pattern* (sys-crate + safe-wrapper, capability traits, codegen),
not code.

**Dependencies:** Conceptually independent; benefits from F1's macro/codegen
patterns being mature enough to reuse the approach.

---

## Suggested rough sequencing

```
M1 ──┬─> F1 (macro codegen) ──> F2 (alphabet coverage) ──> F5 (convert ops)
     │      └─────────────────────> F6 (golden tests, grows with F1/F2)
     ├─> F3 (image-rs boundary)           [independent]
     ├─> F4 (graphynx boundary)           [independent]
     ├─> F7 (release validation)          [type seal did half in M1]
     ├─> F8 (streams/execution context)   [DONE]
     │      ├─> F8.1 (configurable device selection)
     │      └─> F8.2 (async multi-stream chaining)
     └─> F9 (npps signal ops) ── after F1
                F10 (IPP) ── furthest out, reuses the pattern
```

**Sequencing note:** F1, F2, and F8 (core) are complete and merged on `main`.
The cross-cutting F8↔F1 signature-shaping risk (the load-bearing constraint that
shaped the original roadmap) is **resolved**: F8 shipped after F1/F2 with a clean
`_Ctx` regeneration, so the feared "regenerate when streams land" event already
occurred and is closed. All remaining features (F3, F4, F5, F6/F6.1, F7, F8.1,
F8.2, F9, F10) are independent — the next phase is a free choice.

---

## Legend

| Label | Source | Status |
|-------|--------|--------|
| C1–C12 | Architecture review finding (`reviews/final-report.md`) | Various — C1, C2, C5, C12, C7, C8 referenced above |
| CT1–CT6 | Contested finding (same report) | CT5 stride concerns and CT1 `set_len` referenced |
| M1 | Milestone 1 ("build again") | Complete; this roadmap catalogs everything after it |
