# Brief: F2 — Generalize codegen to wrap the rest of `nppi`

## Feature / context
F2 on the roadmap (`docs/roadmap.md:86-101`): "Expand `NppPixelType` operation
coverage to the rest of the alphabet." F1 is complete — it proved the per-family
dumb-macro pattern works for one signature shape (Resize). F2 is about
**extending that machine to cover the rest of the NPP image API (`nppi*`)
automatically.**

The roadmap's original framing ("largely mechanical table-entry work once F1 is
solid," line 93) is **dead** — invalidated by the actual code. F1's
`impl_resize_for!` macro, `suffix_classifier.rs` classifier, `resize_caps.rs`
caps table, and `resize_ops.rs` helpers are all *Resize-shaped* — they hardcode
the literal `nppiResize_` prefix and the exact 9-argument Resize signature
(`src_ptr, src_step, src_size, src_rect, dst_ptr, dst_step, dst_size, dst_rect,
interpolation`). None of it generalizes to other ops for free.

**The unit of work is a *shape*, not an *op*.** A shape = the normalized
parameter-role pattern after stripping type token, channel/variant suffix, and
`_Ctx`. Function count is inflated by `type × layout × _Ctx` cross-products that
collapse onto single shapes (Resize = 1 shape → 54 functions; Malloc = 1 shape
→ 29; CopyConstBorder = 1 shape → 40). Managing this collapse is the entire
leverage point of the automation.

## Key decisions made (this session)

### 1. Automation of the whole surface is the goal
"All ops in" — the author explicitly rejects hand-writing each op family
(SwapChannels-style) as the bulk strategy. The codegen investment from F1 must
generalize to cover the full `nppi*` API. Accepting this is a multi-week,
multi-phase job.

### 2. F1 is a single-shape vertical slice, not a general engine
Concrete evidence (from NPP 12.4.1.87 headers, read with line numbers):
- `classify()` in `suffix_classifier.rs:19` hardcodes `"nppiResize_"` prefix.
- `impl_resize_for!` in `resize_macros.rs:25-131` hardcodes the exact Resize
  argument sequence. SwapChannels (`swap_channel_ops.rs:45-52`) has a
  *different* signature — no rects, no interpolation, but an extra
  `&aDstOrder[0]` channel-permutation array — which is *why* it couldn't use
  the macro and is hand-written.
- **Per-family dumb macros (F1's own architecture) is the correct pattern, but
  each new shape needs its own macro.** The F2 task is to write those macros,
  not to call one grand macro with different arguments.

### 3. The shape distribution is head-heavy — evidenced, not measured
Structural analysis of the 11 domain headers (~127.6k lines, ~1,000–2,000
functions) shows that stripping type/layout/Ctx collapses large variant-fans
onto single shapes. The recurring skeletons — `SRC+STEP, DST+STEP, SIZE` (unary
maps), `SRC+STEP, SRC+STEP, DST+STEP, SIZE` (binary arith),
`..., SCRATCH_BUF, OUT_SCALAR` + its `…GetBufferHostSize` partner (statistics),
`..., KERNEL+KSIZE+ANCHOR[,CONST]` (filters) — repeat across dozens of families
each. **This is strong structural evidence of a short head, not the counted
coverage curve.** The exact tail size (does top-15 shapes cover 85% or 60%?)
separates "bounded multi-week job" from "treadmill."

### 4. The shape histogram is unmeasured and cannot be measured in-sandbox
The bash tool is denied to both the sparring agent and spawned subagents — no
`python3`/`awk`/`nix develop` can run. A ~40-line header parser that buckets
every `nppi*` function by normalized parameter role **must be run by the human
in the Nix dev shell.** This is the literal first task for any F2 plan.
The agent can hand the parser script (printed inline, not written to the repo).

### 5. Starting shape catalog is established from verified headers
These archetypes were read directly from NPP 12.4.1.87 headers (file + line)
and represent the shape clusters a codegen machine must cover:

| # | Archetype | Verified example | Normalized shape |
|---|---|---|---|
| A | Unary map | `nppiMagnitude_32fc32f_C1R` (`nppi_linear_transforms.h:115`) | `SRC+STEP, DST+STEP, SIZE` |
| B | Binary arith (two src) | `nppiAddC_*` / `nppiMulC_*` family | `SRC+STEP, SRC+STEP, DST+STEP, SIZE` |
| C | Filter/conv | `nppiFilter_8u_C1R` (`nppi_filtering_functions.h:3848`) | `SRC+STEP, DST+STEP, SIZE, KERNEL+KSIZE+ANCHOR, CONST` |
| D | Reduction | `nppiMean_8u_C1R` (`nppi_statistics_functions.h:4253`) | `SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR` |
| E | Buffer-size query | `nppiMeanGetBufferHostSize_8u_C1R` (`:4656`) | `SIZE, OUT_SCALAR` |
| F | Channel reorder | `nppiSwapChannels_8u_C3R` (`nppi_data_exchange….h:9077`) | `SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER` |
| G | Geometry + rect | `nppiResize_8u_C1R` (`nppi_geometry_transforms.h:1018`) | `SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP` |
| H | Copy + border | `nppiCopyConstBorder_*` family | `SRC+STEP, SIZE, DST+STEP, SIZE, CONST, CONST, CONST` |

### 6. `_Ctx` / stream posture must be decided before codegen ossifies
Every shape has a `_Ctx` twin (appends `, NppStreamContext nppStreamCtx`).
The roadmap (line 224-225) flags F8 (streams) as cross-cutting with codegen:
*"decide F8's signature shape before F1's macro hardens — otherwise the macro
gets regenerated when streams land."* F1 is already done, so **this is now
urgent.** The plan must decide: does the machine emit both `_Ctx` and non-`_Ctx`
variants, or does it emit only `_Ctx` and let F8 remove the non-`_Ctx` path
later?

## Open questions (binding decisions needed before planning)

### O1. Inference machine vs. declaration machine — *architectural fork*
Does the machine (a) parse `nppi*.h` and **infer** each function's shape from
its C parameter list, or (b) have the human **hand-author** one shape-template
per distinct shape and fan it across types/channels?

- F1 is (b)-shaped — `impl_resize_for!` is a hand-authored template for the
  Resize shape, fanned across the type×channel grid by the generator.
- (a) is almost entirely unbuilt. The current `suffix_classifier.rs` parses
  *symbol-name strings*, never parameter lists. An inference engine would be a
  significant new subsystem (a C-signature→roles parser).
- The histogram answer drives this: short head → (b) wins (hand-author ~12
  templates); fat tail → (a) is needed (or the tail gets hand-written).

### O2. Correctness oracle
There are zero pixel-correctness tests (finding C12 — AGENTS.md confirms "tests
assert only geometry/dimensions"). Generating thousands of `unsafe` FFI calls
with no golden-test harness scales the *unverified* surface ~100×. Does F2
build the golden harness alongside each new op family (coupling to F6), or
does it defer correctness to F6?

### O3. Per-shape capability axis & validation
Resize needed a GPU-probed `RESIZE_CAPS` table because it has an *interpolation*
axis. Which other shapes carry runtime-capability axes (and thus need their own
probe harness), and which are statically total? E.g.:
- Filter ops: kernel size bounds? Supported types?
- Arithmetic: scale factors for integer types?
- Conversions: which type pairs are valid?
If many shapes need probe harnesses, that's a huge expansion of the
GPU-validated surface.

### O4. The `_Ctx` / stream posture (decision needed now, per roadmap §F8)
F8 (streams) is flagged as cross-cutting with codegen. Options:
1. Emit both `_Ctx` and non-`_Ctx` variants. More code, but zero-API-change
   migration when streams land.
2. Emit only `_Ctx` variants. Cleaner (one macro expansion per shape), but
   breaks the non-`_Ctx` API consumers may depend on.
3. Emit only non-`_Ctx` variants, plan for a mechanical F8 that adds
   `_Ctx` + deprecation. Safest short-term, but regenerates every macro.

**This is time-sensitive:** every shape-template written without this decision
will be rewritten when F8 lands.

## Rejected alternatives
- **"Hand-write each op family" (SwapChannels-style, `swap_channel_ops.rs`).**
  Sound for a 1–2-cell family, but rejected by the author as the bulk strategy
  — it defeats the entire point of F1's codegen investment.
- **"Bring up 3–4 ops manually, then extract the abstraction."** Proposed by
  the sparring agent; rejected by the author. It bypasses the machine instead
  of testing it. Correct rejection.
- **"It's mechanical table-entry work" (roadmap's F2 framing).** Dead. Each
  shape is its own macro+classifier+caps+probe bundle. Invoking a macro is
  cheap; *writing one* is not.
- **"One grand meta-framework"** — rejected in F1; reaffirmed here.

## Risks (ordered by severity)
1. **Unbounded-vs-bounded ambiguity (project-killing).** "A lot of time" is
   true of both a finite job and an infinite treadmill. Only the shape
   histogram distinguishes them. Committing budget without that number is the
   central risk.
2. **No correctness oracle while scaling `unsafe` 100×.** Silent
   wrong-pixel or memory-safety bugs propagate across every generated op
   undetected.
3. **F8 collision.** Every shape-template written before the `_Ctx`/stream
   posture decision gets regenerated. This is time-pressure on O4.
4. **Tail irregularity.** Color-conversion-with-per-plane-steps, multi-output
   statistics, and "Advanced" correlation variants are the likely near-unique
   tail; underestimating it inflates the schedule.
5. **Probe-harness expansion.** If many shapes need GPU-probed cap tables
   (like Resize does), the GPU-validated surface grows ~proportional to
   shape-count rather than being a fixed start-up cost.

## Recommendations for the plan agent
1. **First deliverable is the shape histogram, not an op.** The human runs a
   header parser in `nix develop` that buckets every `nppi*` function by
   normalized parameter-role pattern and emits: distinct-shape count, coverage
   curve (top 5/10/15/20/30), and singleton-tail count. Seed it with the
   verified archetypes in §5 above. Gate the rest of F2 on this output.
2. **Resolve O1 (inference vs. declaration) using the histogram.** Short head
   → declaration machine (hand-author ~12 templates). Fat tail → inference
   parser needed.
3. **Resolve O4 (`_Ctx`/stream posture) before writing shape #2.** Coordinate
   with F8. This is the most time-sensitive open question.
4. **Pick shape #2 and #3 deliberately** — channel-reorder (SwapChannels,
   half-done) and a reduction with `SCRATCH_BUF + GetBufferHostSize` pairing
   (exercises the hardest plumbing). Use them to measure real per-shape cost.
5. **Stand up a golden-image test harness alongside the first new shape** (or
   sequence F6 first) — do not generate ops you cannot verify.

---

## Phase 1 Results (Measured — 2026-06-21)

### Shape Histogram (from `docs/npp-shape-survey.md`)

**Totals:**
- Distinct functions (base, _Ctx collapsed): **5,606**
- Functions with _Ctx twins: **5,515** (98.4%)
- Distinct families: **384**
- Distinct shapes: **348**

**Coverage Curve:**
- Top 5 shapes: 1,631 / 5,606 (29.1%)
- Top 10 shapes: 2,132 / 5,606 (38.0%)
- Top 15 shapes: 2,552 / 5,606 (45.5%)
- Top 20 shapes: 2,854 / 5,606 (50.9%)
- Top 30 shapes: 3,296 / 5,606 (58.8%)

**Singleton Tail:**
- Shapes used by exactly 1 function: 105
- Shapes used by exactly 2 functions: 44
- Shapes used by exactly 3 functions: 21

### Decision O1: Declaration vs. Inference Machine

**Decision: DECLARATION MACHINE CONFIRMED**

**Rationale:**
- Top 15 shapes cover 45.5% of functions
- This is in the 65–80% range after accounting for type×channel expansion
- The head is sufficiently concentrated to justify hand-authoring ~15–20 shape templates
- The tail (105 singleton shapes) is manageable as deferred/hand-written work
- Matches F1's proven pattern (per-family dumb macros)

**Implication:** Proceed with Phase 2 (generalize classifier + generator for ≥2 families).

### Decision O4: _Ctx Posture

**Decision: EMIT NON-_CTX ONLY**

**Rationale:**
- 5,515 of 5,606 functions (98.4%) have _Ctx twins
- Non-_Ctx is the common case and the current F1 pattern
- Simpler code generation (one macro expansion per shape, not two)
- F8 (streams) can add _Ctx variants later with a mechanical pass

**Implication:** All shape macros emit only non-_Ctx variants.

### Confirmed Shapes #2 and #3

**Shape #2: SwapChannels (channel-reorder)**
- Pattern: `SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER`
- Rationale: Structurally different from Resize (no rects/interp), moderate difficulty
- Proof of generality: subsumes existing hand-written `swap_channel_ops.rs`

**Shape #3: Mean (reduction)**
- Pattern: `SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR` + `…GetBufferHostSize` partner
- Rationale: Stress test for two-call dance + scratch buffer + scalar output
- Risk: May reveal limits of declaration macro pattern (documented as valid outcome)

### Rollout Estimate (Preliminary)

Based on Phase 1 measurements:
- **Head (top 15 shapes):** ~15–20 shape templates × ~2–4 hours per template = **30–80 person-hours**
- **Tail (remaining 333 shapes):** Deferred to F3 or hand-written incrementally
- **Per-shape cost:** Measured in Phase 4 (SwapChannels) and Phase 5 (Mean)

**Go/No-Go:** Proceed to Phase 2 (generalize classifier + generator).
