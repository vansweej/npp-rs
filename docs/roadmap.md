# npp-rs Roadmap

> Work after Milestone 1 ("build again on latest CUDA via Nix + cudarc").
> Each feature is a candidate for its own sparring and planning session.
> Prioritization and sequencing will be decided per session; this doc is
> a descriptive catalog, not a commitment schedule.

---

## F1 — Macro-generated binding codegen *(recommended first after M1)*

**What:** Replace the hand-written `u8`/`f32` capability-trait impls from M1 with
macro-generated impls covering the full `NppPixelType` alphabet.

**Why:** M1 deliberately hand-writes two exemplars (`u8`, `f32`) precisely so
this milestone has a proven, trait-shaped target to generalize from. The macro's
job is "emit exactly what the M1 hand-written impls look like, parameterized
over type/channel/op."

**Known hard problems:**
- *Irregular NPP symbol grid.* Naming has multiple axes that don't all compose:
  element type (`8u`/`32f`/…), channel count (`C1`/`C3`/`C4`),
  channel-changing ops (`C4C3R` for `bgra_to_rgb` — breaks a single-`C<n>`
  template), ROI vs scaled (`R` vs `Sfs`), stream-context variants. The macro
  can't string-interpolate a suffix; it must select from a hand-curated table
  of which symbols actually exist.
- *Sparse capability matrix.* Even within one op, `(type × mode)` is sparse
  (e.g. `16f` does not support Lanczos resize). "Type supported" ≠ "all modes
  supported."
- *Verification must be compile-time, not link-time.* The bindgen `nppi*`
  allowlist output (`bindings.rs`) should be the source of truth: generate calls
  only over symbols bindgen actually emitted, so a wrong table entry is a
  `npp_sys::` "item not found" compile error — not an `undefined reference` at
  the end of a multi-minute link.

**Dependencies:** Hard-depends on M1 (needs the proven trait exemplars and the
validated FFI pointer bridge).

---

## F2 — Expand `NppPixelType` operation coverage to the rest of the alphabet

**What:** Once F1's macro works, fill in `Resize`, `SwapChannels`, and other
capability traits for `16u/16s/16f/32u/32s/32f/64f/8s` wherever NPP provides
the symbol.

**Why:** M1 ships the *alphabet* (all types constructible) but only `u8`/`f32`
*ops*. This closes the gap — driven by the macro, so it should be largely
mechanical table-entry work once F1 is solid.

**Note:** Partly fused with F1 in practice (the macro and coverage grow
together), but worth tracking separately because "macro works for 2 types" and
"macro covers all real NPP cells" are different definitions of done.

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
| M1 | Milestone 1 ("build again") | In progress; this roadmap catalogs everything after it |
