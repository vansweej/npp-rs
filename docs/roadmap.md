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
- C8 — stream/execution-context model (CUDA streams, async ops)
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

## F5 — Pixel-format conversion ops (`u8 ↔ f32`, scaling)

**What:** On-device pixel *conversion* operations — e.g.
`RgbImage<u8> → RgbImage<f32>`, normalization/scaling — backed by NPP's
`nppiConvert_*` / `nppiScale_*` families.

**Why:** A convert is its own NPP op, slotting in as another capability trait
once the macro infrastructure exists. The `u8→f32` normalise is the exact
operation an NN-preprocessing pipeline wants, so it may deserve priority once
the alphabet ops land.

**Dependencies:** Cleanest after F1/F2 (it's "just another capability" once
codegen exists), but the normalise use case may justify hand-writing
`Convert for CudaImage<u8> → CudaImage<f32>` needed-earlier.

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

## F8 — Stream / execution-context model (C8 + C7)

**What:** A first-class stream/execution-context abstraction: configurable
device selection (kill the hardcoded `Device::get_device(0)`), explicit stream
assignment, defined synchronisation contract per operation, and compute/copy
overlap.

**Why:** C8 — there is currently *zero* stream concept; correctness is an
emergent side-effect of default-stream read-back ordering. This forecloses
async overlap (the entire performance reason to use CUDA asynchronously) and
creates latent races the moment two ops are chained. cudarc has the primitives
(`fork_default_stream`, `wait_for`, `CudaStream`), so this is "design the
abstraction over cudarc's stream API," not "build streams from scratch."
Signature-shaping — it touches every op signature (which stream?), so it wants
its own ADR/sparring session before it's wired in.

**❗ Cross-cutting flag:** F8 interacts with F1 (macro). If streams are coming
and they change op signatures, decide F8's signature shape *before* F1's macro
hardens — otherwise the macro gets regenerated when streams land.

**Dependencies:** M1.

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
     ├─> F3 (image-rs boundary)           [independent of F1]
     ├─> F4 (graphynx boundary)           [independent of F1]
     ├─> F7 (release validation)          [type seal did half in M1]
     ├─> F8 (streams/execution context)   [decide signature shape *before* F1]
     └─> F9 (npps signal ops) ── after F1
                F10 (IPP) ── furthest out, reuses the pattern
```

The one cross-cutting dependency worth remembering: **F8 (streams) interacts
with F1 (macro).** If streams are coming and they change op signatures, decide
F8's signature shape *before* the F1 macro ossifies — otherwise the macro gets
regenerated when streams land. Everything else is comfortably independent.

---

## Legend

| Label | Source | Status |
|-------|--------|--------|
| C1–C12 | Architecture review finding (`reviews/final-report.md`) | Various — C1, C2, C5, C12, C7, C8 referenced above |
| CT1–CT6 | Contested finding (same report) | CT5 stride concerns and CT1 `set_len` referenced |
| M1 | Milestone 1 ("build again") | Complete; this roadmap catalogs everything after it |
