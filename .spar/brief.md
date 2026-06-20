# Brief: F1 architecture — per-family dumb macros, two-pipeline codegen

## Feature / context
F1 on the roadmap (macro-generated binding codegen). The housekeeping that
unblocked it (byte-step unification, M1-plan retire, AGENTS.md sync) was
completed in commits `c818f3c`–`1a09618`. Now designing the architecture
itself.

## Key decisions made (this session)

### 1. Per-family dumb macros, not a unified meta-framework
NPP's own per-family signatures are too irregular for a single framework to
abstract over. No string-interpolation of symbol names; emit dumb
per-family `macro_rules!` that take the symbol and type as data, and expand
to exactly the body the exemplars (now unified on byte-step form) already
show. The macro is a **boilerplate reducer**, not an abstraction layer.

### 2. Resize-only scope for the first macro
The Resize grid is genuinely wide (~8 types × 3 channel layouts) — the macro
pays for itself immediately. SwapChannels stays hand-written for now because
it has only one symbol (`nppiSwapChannels_8u_C4C3R`), so a macro would be a
net complexity cost until more symbols exist.

### 3. Two derivation pipelines for the two axes of capability
This is the core architecture. F1 must emit code that is parameterized over
two independent axes, and they are derived from fundamentally different
sources:

| Axis | What | Source | Regeneration trigger |
|------|------|--------|----------------------|
| `type × channel` | Which NPP symbols exist for a given `(T, C)` | Scraped from bindgen output (`bindings.rs`) | Every build (bindgen regenerates) |
| `type × interpolation` | Which interpolation modes are supported for a given `(T, mode)` | Probed from NPP at runtime on a GPU box | Human-initiated on CUDA bump; output committed |

### 4. Scrape `type × channel` from `bindgen` output
`bindings.rs` (gitignored, regenerated every build) already names every NPP
symbol. A **suffix classifier grammar** — a small, hand-maintained set of
~5 rules — selects `C1R`/`C3R`/`C4R` from a symbol suffix and rejects
`AC4`, `P{n}`, `SqrPixel`, `Batch`, and other siblings that don't fit the
simple channel-layout grid. This classifier is the *only* hand-maintained
artifact on the channel axis. It lives alongside the probe output as
committed source consumed at `macro_rules!` expansion time.

### 5. Probe `type × mode` from NPP at runtime, not from prose
The interpolation-mode validity matrix (`16f` ✗ Lanczos, etc.) **cannot** be
scraped from function names — interpolation is a runtime `int`, not a symbol
suffix. The library itself is the ground truth. A human-initiated GPU probe
(a test harness run on a CUDA bump) exercises every `(type, mode)` cell and
records which return success. Probe output is **committed to the repo** so
CI and contributors (who have no GPU) have it at compile time.

### 6. Probe output is a committed source artifact; `bindings.rs` is gitignored
This is the inverse of the `bindings.rs` policy, and the asymmetry is
*correct* for each artifact's regeneration constraint:
- `bindings.rs` can be regenerated every build (CUDA SDK on the build
  machine) → gitignored.
- Probe output needs a GPU to produce and no GPU on CI → committed.
- **Must be loudly documented** so contributors don't instinctively
  `gitignore` the probe output, mistaking it for another generated artifact.

### 7. Status-code spike is the gating prerequisite to the probe
The probe must distinguish "mode unsupported" from "harness bug" (image too
small for a kernel footprint) — otherwise a too-small probe image silently
marks a valid cell as invalid. The spike runs on a GPU box and must produce
a **closed taxonomy**:

| `NppStatus` | Label | Probe action |
|---|---|---|
| `NPP_SUCCESS` (≥ 0) | supported | emit `true` for cell |
| `NPP_INTERPOLATION_ERROR` (-22) | mode unsupported | emit nothing (cell = `false`) |
| `-201` (pinned) | probe image too small | **FAIL LOUD** — do not emit a row |
| anything else | unknown | **FAIL LOUD** — spike was incomplete |

The spike is kept as a committed CUDA-bump regression guard (not throwaway).
The negative case uses an invalid interpolation value (999) since `16f` has no
`nppiMalloc` allocator and `8u+Super`/`8u+Lanczos` are both supported on CUDA 12.9.
The three pinned codes: positive=0, negative=-22, harness-bug=-201.

Prove by constructing three test cases:
- **Known positive:** `f32 × Lanczos` (fully supported) → confirm `NPP_SUCCESS`.
- **Known negative:** `8u × invalid-interpolation(999)` → capture the
  exact `NppStatus` that means "unsupported" (NPP_INTERPOLATION_ERROR = -22).
- **Deliberate harness bug:** tiny image (1×1) × Lanczos → confirm
  the error code (-201) is *different* from the unsupported code.

After the spike, the probe internally converts every `NppStatus` via this
table and aborts on anything not in the taxonomy.

## Rejected alternative: LLM-extraction of the interpolation matrix
Having an LLM read NPP's prose docs and emit a capability table was the
initial instinct but was rejected: it introduces a human-interpretation
error mode (the LLM can hallucinate cells) for which the only validator is
NPP itself. Runtime probing removes the interpreter entirely: NPP answers
directly, with zero hallucination risk.

## Rejected alternative: one grand macro framework
A unified meta-framework that discovers symbols, generates traits, and
emits impls was rejected in favor of per-family dumb macros. The meta-
framework would need to understand every op family's signature variation,
which is equivalent effort to maintaining per-family macros, with worse
local reasoning.

## Rejected alternative: runtime `NotImplemented` errors
Every `(type, mode)` that is emitted compiles **only** if the probe confirmed
it is valid. No runtime "not implemented" path — the build graph itself
encodes the capability matrix.

## Artifact inventory (new/changed)

| Artifact | Status | Regeneration | Committed? |
|----------|--------|-------------|------------|
| `npp/src/resize_macros.rs` | **new** | Per-family `macro_rules!` | yes |
| `npp/src/suffix_classifier.rs` | **new** | Hand-maintained ~5-rule grammar | yes |
| `npp/src/resize_caps.rs` | **new** | Probe output (GPU needed) | **yes** (committed) |
| `npp/src/resize_generated.rs` | **new** | Generator example output | **yes** (committed) |
| `npp/tests/fixtures/nppiResize_symbols.txt` | **new** | Captured from one Nix build on a CUDA host | **yes** (committed) |
| `npp-sys/src/bindings.rs` | existing | Bindgen every build | **no** (gitignored) |
| `npp-sys/build.rs` | existing | Scraper entry point | yes |

Channel count is **runtime data**, not a type parameter — `CudaImage::new(device, channels: u8, …)`,
stored in `layout.channels`. The macro dispatches via `match self.channels()` to C1/C3/C4 symbols.
`16f` cannot be driven from the safe layer (`half` crate disabled); it is probed via raw `npp_sys` FFI
and skipped in the generator. The mode-safety question (decision #7 in the original brief) is resolved
as **runtime-checked** against the committed `RESIZE_CAPS` table, not compile-time.

The suffix classifier and probe output are the two committed sources the
macro consumes. `bindings.rs` is consumed indirectly (the scraper reads it,
the macro reads the scraper's output).

A THIRD committed artifact exists purely for *testing*: a frozen corpus of
real `nppiResize_*` symbol names. The classifier consumes the live
(gitignored) `bindings.rs` at build time, but its *unit tests* run offline
against this committed corpus — so the grammar can be validated with plain
`cargo test`, no CUDA, no GPU.

## Open questions (need the spike to answer)
1. Does the unsupported code differ between type-level rejection (`16f` at
   all) vs. mode-level rejection (Lanczos specifically)? The spike on
   `16f × Lanczos` plus `16f × Linear` answers this.
2. What is the minimum probe image size such that every *supported* mode
   returns success? Determined by finding the largest kernel footprint
   (Lanczos, Super) and allocating marginally above it.
3. (Resolved) Where does the scraper / suffix classifier live? **Neither** —
   the classifier is a plain Rust module: `fn classify(symbols: &[&str]) -> Vec<ClassifiedSymbol>`.
   No build script, no OUT_DIR dependency. Its unit tests consume the
   committed fixture (`nppiResize_symbols.txt`); the actual call site in a
   build script (or wherever the macro-produced impls live) is responsible
   for reading `bindings.rs` and passing symbol names into it. The
   classifier itself is pure parsing — no cargo coupling.

## Status (post-implementation)
All items 0–6 above are **implemented**. The `impl_resize_for!` macro is defined
in `resize_macros.rs`, the generator lives at `examples/gen_resize_impls.rs`, and
the committed output is `src/resize_generated.rs`. The suffix classifier module
(`suffix_classifier.rs`), probe table (`resize_caps.rs`), and spike spike
(`tests/spike_npp_status.rs`) are all committed. Channel dispatch is runtime
(match on `CudaImage::channels()`). Mode safety is runtime (`mode_supported`
against `RESIZE_CAPS`). See `docs/roadmap.md` for the final architecture.
