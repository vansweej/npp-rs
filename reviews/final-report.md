# Final Architectural Synthesis — `npp-rs`

**Artifact:** `npp-rs` workspace (`npp-sys` FFI bindings + `npp`/`npp-rs` safe wrapper) — Rust bindings to NVIDIA NPP image-processing primitives, CUDA 10.2, `0.0.1`/pre-alpha.
**Inputs synthesised:** 8 architect reports across 2 rounds (principal, design, complexity, security).
**Synthesis date:** 2026-06-19
**Moderator role:** Synthesis only. The verdicts below reflect cross-panel consensus weighted per the rules; they are not the moderator's independent opinions.

> **Devil's-advocate handling.** `arch-complexity` was the designated devil's advocate in Round 2. Its Round 2 report is a set of adversarial stress-tests, not sincere positions. Where its Round 2 challenges are cited below, they are treated as falsifiable hypotheses ("what evidence would change the conclusion") rather than as consensus dissent. Its *Round 1* findings are weighted normally.

---

## Executive Summary

The panel is unanimous and stable across both rounds: the **macro-architecture is sound** (the `-sys`/safe-wrapper split, the `CudaLayout` geometry abstraction, and `image`-crate interop via `TryFrom` are the correct skeleton), but the crate is **not production-ready**, and three architects rendered an explicit reject/"not production-ready" disposition. The decisive issues are a small, concrete set of **API-signature-shaping decisions** — a thread-hostile ownership model (`Rc<RefCell<…>>`), the total absence of any stream/execution-context model, and an error type that collapses every failure to `CudaError::UnknownError` — plus **one genuinely reachable release-mode memory hazard**: format/stride preconditions guarded only by `debug_assert!`, which compile out in `--release`. The most important outcome of Round 2 is *subtractive*: peer cross-examination (including the design and principal architects retracting their own claims) demolished two of the original "Critical" rejection gates — a claimed memory leak and a claimed out-of-bounds bounds-check bug — redirecting remediation effort away from non-existent defects and toward the genuine ones. Overall signal: **structurally promising, internally unsafe in release, and cheap to fix now / expensive to fix after the API ossifies.**

---

## Confirmed Findings

Findings independently raised or explicitly endorsed by three or more architects. Each verdict reflects post-Round-2 consensus.

### C1 — Error collapse to `CudaError::UnknownError` destroys operability — **Confirmed**
Every non-zero NPP status and every `image`-layer failure is mapped to a single opaque `CudaError::UnknownError` (`resize_ops.rs:91`, `swap_channel_ops.rs:47`, `image.rs:187`). This conflates three distinct error domains (NPP / CUDA driver / image codec), discards NPP's rich signed `NppStatus`, and renders production failures undiagnosable.
- **Confirmed by:** all four personas — arch-principal (§4.3), arch-design (R5), arch-security (OBS-02), arch-complexity (R1 finding 3). Four-way convergence with zero dissent across both rounds; arch-principal flagged it as the lowest-controversy, highest-agreement gate. The devil's-advocate Round 2 challenge argued only that *fixing it now vs. later* is a sequencing question (because `NppStatus` is CUDA-version-specific), not that the defect is unreal — so the finding itself stands unchallenged.
- **Recommended action:** Introduce a dedicated error enum that preserves the underlying `NppStatus` integer and keeps NPP / CUDA / image variants distinct, before the API surface grows (the change is breaking if deferred). Note the inverse-direction corollary in C2.

### C2 — Safety preconditions are `debug_assert!` and vanish in release builds — **Confirmed (primary reachable memory hazard)**
The only guards on format/stride/dimension invariants feeding the `unsafe` NPP calls are `debug_assert!` (`resize_ops.rs:38-39`; `swap_channel_ops.rs:9-13`), which compile to nothing under `--release`. Because `CudaImage::new(w, h, ct)` accepts an arbitrary `ColorType` (`image.rs:31`), a caller can construct a non-3-channel `CudaImage<u8>` and pass it to `resize`, which unconditionally calls `nppiResize_8u_C3R`. Worked example (arch-principal §3.1, Round 2): a 1-channel buffer is `w*h` bytes; C3R reads ~3 bytes/pixel, reaching ≈ `w*h + 2w - 3` — past the allocation end. This is reachable from safe code in the default release profile and is the **real** out-of-bounds path.
- **Confirmed by:** arch-security (RISK-02, HIGH), arch-complexity (R1 finding 3), arch-design (R2/R4 → promoted in Round 2 to its *primary* gate), arch-principal (endorsed in Round 2 §3.1 as the hazard it under-weighted in Round 1).
- **Devil's-advocate stress-test (weighted as a challenge, not consensus):** arch-complexity's Round 2 audited the public constructors and argued the typed `TryFrom` paths (`RgbImage`→3ch, `RgbaImage`→4ch) cannot produce a wrong-channel image, so the asserts guard invariants the constructors already enforce. **Partial rebuttal grounded in source:** this holds for the `TryFrom` paths, but **not** for `CudaImage::new`, which takes an open `ColorType` and is the documented escape. The devil's-advocate's own preferred remedy (seal `T`/channels at the type level) and the consensus remedy converge on the same place.
- **Recommended action:** Replace `debug_assert!` with `Result`-returning validation (or `assert!`) at the FFI boundary **and** seal the type so wrong-format images are unconstructible (newtype `CudaImageRgb8` / `NppPixelType` marker trait, removing or bounding the generic `T`). Doing both closes the class at compile time and at the boundary.

### C3 — `Rc<RefCell<DeviceBuffer<T>>>` makes `CudaImage` thread-hostile — **Confirmed (as a blocker for the stated domain), with one open dependency** (see U1)
`CudaImage` holds `Rc<RefCell<DeviceBuffer<T>>>` (`image.rs:14`). `Rc` is `!Send + !Sync`, so the type cannot move to a loader thread, be shared across a `rayon` pool, or be held across a `tokio` await — the exact usage the stated NN-preprocessing domain implies. The choice exists only to make `sub_image` aliasing cheap (`Rc::clone`, `image.rs:108`), trading total cross-thread usability for single-thread convenience. The decision is encoded into every operation signature, so its cost grows with each primitive added.
- **Confirmed by:** arch-principal (§4.1, HIGH), arch-design (R7 in Round 1, then explicitly upgraded in Round 2 §3.1 — withdrawing its own "acceptable for now"), arch-security (RISK-07-adjacent / Round 2 §3.2), and arch-complexity's Round 1 finding 1 (which identified the `RefCell` half of the problem).
- **Decomposition the panel converged on (arch-principal Round 2 §1.2).** Three separable concerns, each with a different verdict:
  - *Cross-thread transfer* — `Rc` is `!Send` → **the real blocker** (needs `Arc` or an owned-buffer design).
  - *Interior mutability* — `RefCell` is **pure overhead**; `as_device_ptr()` needs only `&self`, and the `borrow_mut()` calls are statement-scoped temporaries that never guard the FFI call (Confirmed — see C4).
  - *Aliased mutation* — intrinsic to the chosen model; `RefCell` does **not** catch it (see C4).
- **Recommended action:** Resolve via ADR **before** wrapping more primitives. Two viable directions: (a) `Arc<Mutex/RwLock<…>>` (restores `Send`/`Sync` at the cost of atomic refcount + locking), or (b) an owned-buffer `CudaImage` (`Send`) plus a borrowed `CudaImageView<'a>` that pushes aliasing into the type system. **Caveat:** the devil's advocate raised a substantive precondition on remedy (a) — see U1 — that must be checked first.

### C4 — The `RefCell` provides no safety and masks a real src/dst aliasing hazard — **Confirmed**
In every operation the `RefMut` from `borrow_mut()` is a per-statement temporary dropped at the `;` before the raw pointer reaches NPP (`resize_ops.rs:61-74`; `swap_channel_ops.rs:19-32`; `image.rs:170-177`). Therefore the `RefCell` cannot detect or prevent two raw device pointers into the *same* buffer being passed as `src` and `dst`. `test_resize3` (`resize_ops.rs:177-178`) exercises exactly this: two `sub_image` views of one parent buffer. The ROIs there happen to be spatially disjoint (so NPP is well-defined), but nothing in the type system enforces non-overlap, and overlapping in-place resize is undefined per NPP.
- **Confirmed by:** arch-complexity (R1 finding 1 — the redundancy), arch-design (R7 — the aliasing), arch-principal (Round 2 §3.3 — the synthesis of the two), arch-security (Round 2 §3.3 + NEW-03 — confirmed mechanically, plus the false-audit-assurance angle).
- **Round 2 self-correction folded in:** arch-design retracted its Round 1 claim that the two `borrow_mut()`s cause a double-borrow panic and that `test_resize3` uses *overlapping* ROIs — both inaccurate on inspection. The corrected mechanism (sequential temporaries; disjoint test ROIs; unenforced contract) is what is Confirmed here.
- **Recommended action:** Remove `RefCell`. Document or type-enforce the src/dst non-aliasing contract for `resize`/`bgra_to_rgb`. Note the remedy is **not** `Rc::get_mut` (see Contested O1 — it cannot work in the aliasing case). The container choice should fall out of the C3 ownership ADR, not be reverse-engineered in isolation.

### C5 — `Vec::with_capacity` + `set_len` exposes uninitialised memory in safe `RgbImage::try_from` — **Confirmed (defect), severity Contested** (see CT1)
`RgbImage::try_from(&CudaImage<u8>)` allocates with `Vec::with_capacity(size)` then `set_len(size)` before the device→host copy fills it (`image.rs:161-163`), reachable from a safe public API. The pattern is unsound practice and must be replaced.
- **Confirmed by:** arch-design (R1), arch-security (RISK-04, elevated in Round 2 §1.3), arch-principal (Round 2 §2.6, acknowledged as a real smell), arch-complexity (Round 2 — acknowledged as real but "dormant").
- **Recommended action:** Use zero-initialised allocation or `MaybeUninit` with a verified full-write; make the read-back stride-correct for sub-images; add a byte-exact round-trip test for full images **and** ROIs. (Whether it is *currently reachable* UB is Contested — CT1 — but all four agree the pattern must change regardless.)

### C6 — CI compiles but never tests; runs on a retired runner; no GPU lane — **Confirmed**
`.github/workflows/build.yml` runs `cargo build` only on `ubuntu-18.04` — no `cargo test`, `clippy`, or `fmt`, and the GitHub runner has no GPU, so the safety-critical FFI/pointer-arithmetic tests never execute in CI. `ubuntu-18.04` runners are retired, so the pipeline is also failing for infrastructure reasons. No `flake.nix` exists, so the project's intended Nix reproducibility story is aspirational.
- **Confirmed by:** all four — arch-principal (§4.5), arch-design (R9), arch-complexity (R1 finding 7), arch-security (OBS-05). arch-design added a sharper Round 2 corollary (§5.3): because `npp` depends on `npp-sys = "0.0.1"` from the registry rather than the local path (`npp/Cargo.toml:13-14`), CI compiles against the *published* bindings, not the in-tree `npp-sys` under review — so the one thing CI does may not even exercise the local FFI surface.
- **Recommended action:** Migrate off `ubuntu-18.04`; add `clippy -D warnings`, `fmt --check`, and `cargo test`; stand up a GPU-capable lane (self-hosted runner or documented manual gate) so the device tests actually run; repoint `npp-sys` to the workspace path for development. The devil's advocate concedes this is "the one genuine, undebatable defect."

### C7 — `CUDA Context` lifetime is an unenforced, untyped invariant — **Confirmed**
`initialize_cuda_device()` (`cuda.rs:4-8`) returns a `Context` the caller must keep alive; tests park it in `_ctx`. If the context drops while any `CudaImage` (or `sub_image` `Rc` clone) is live, the subsequent `DeviceBuffer::drop` calls `cuMemFree` against a destroyed context — a use-after-free at the CUDA driver layer. The invariant is undocumented and unenforced by types.
- **Confirmed by:** arch-principal (§5 + Round 2 §5.3), arch-design (Round 2 §4), arch-security (Round 2 §3.4). This is the *correct* re-framing of what arch-design Round 1 mislabelled as a "leak" (see CT2).
- **Recommended action:** Tie buffer lifetime to context lifetime in the type system (e.g., a phantom lifetime binding `CudaImage`/`DeviceBuffer` to `Context`). Treat as part of the same execution-context abstraction as the stream model (C8).

### C8 — No stream / execution-context model — **Confirmed**
There is no first-class stream or execution-context concept anywhere (`grep` finds zero `Stream`/`synchronize` references). Every op runs on the default stream with no explicit synchronisation; correctness is an emergent side-effect of the read-back copy being implicitly ordered after the kernel on the default stream. This forecloses compute/copy overlap (the entire performance reason to use CUDA asynchronously) and makes latent races appear the moment two ops are chained with an intermediate read.
- **Confirmed by:** arch-principal (§4.2), arch-design (Round 2 §3.4 — endorsed as a gap it missed), arch-security (Round 2 §3.2 — reclassified its RISK-07 timing concern as subordinate to this correctness gap). arch-principal's Round 2 §5.3 unifies device-selection, context, stream, and teardown into a single missing abstraction (an *owned execution context*) that is the common parent of four separately-reported symptoms.
- **Recommended action:** Introduce a first-class stream/execution-context type and define the synchronisation contract of every operation. This is signature-shaping → settle by ADR before the API grows. Bundle with C7 (context lifetime) and configurable device selection (currently hardcoded `Device::get_device(0)`, `cuda.rs:6`).

### C9 — `Persistable::save` silently redirects to the temp dir and lies about its contract — **Confirmed (as an API-contract footgun); the "security" framing is Contested** (see CT3)
`save(filename)` ignores the caller's path semantics: it pushes `filename` onto `temp_dir()`, forces `.png`, and `.unwrap()`s a `to_str()` that can panic on non-UTF-8 (`image.rs:192-201`). A caller passing `/output/result` writes to `/tmp/result.png`.
- **Confirmed by:** arch-complexity (R1 finding 4 — "the signature lies"), arch-security (RISK-05), arch-principal (§5), arch-design (Round 2 §3.6).
- **Recommended action:** Accept a real `&Path` (and honour it), or rename to reflect the temp-dir behaviour; remove the panicking `unwrap`. Whether it additionally constitutes a *path-traversal vulnerability* is Contested (CT3).

### C10 — EOL/abandoned toolchain pins consumers to a 2019 stack — **Confirmed**
CUDA 10.2 hard-pin with no version abstraction (static link names in `build.rs`, `ubuntu-18.04` CI), `rustacuda 0.1` (unmaintained, no CUDA 11+ support), `bindgen 0.58`, `image 0.23`, `criterion 0.3`. The stated CUDA 11.x goal is not expressible without editing `build.rs`. The active successor to `rustacuda` is `cudarc`.
- **Confirmed by:** all four — arch-principal (§4.4), arch-design (R8), arch-complexity (R1 finding 7 + open questions), arch-security (OBS-01/OBS-05).
- **Recommended action:** Plan a CUDA 11/12 + maintained-binding migration; add a CUDA-version abstraction (feature flags + library-layout handling) and a `bindgen` allowlist; record the decision and timeline in an ADR. (Sequencing relative to the error-type work is itself Contested — see CT4.)

### C11 — Generic `T` is not just unused scaffolding — it is actively unsound off the `u8` path — **Confirmed**
`CudaImage<T>` is generic but every real path is hard-bound to `u8`/3-channel. Worse, `CudaImage::new` folds element size into the channel count: `row_major_packed(size_of::<T>() as u8 * ct.channel_count(), …)` (`image.rs:36`). For `T=u8` the two coincide; for `T=f32` (the type an inference pipeline actually wants for normalised tensors), `channels` becomes `4 × channel_count`, and every channel-based computation silently breaks (`get_start_point` divides `img_index` by `channels`, the assert guards, NPP variant selection). arch-design adds a parallel allocation-sizing concern (`DeviceBuffer::zeroed` counts elements of `T`, not bytes).
- **Confirmed by:** arch-complexity (R1 finding 3 — "false promise"), arch-design (Round 2 §5.1 — "the signature lies"), arch-principal (Round 2 §5.2 — "actively wrong"). The devil's advocate (arch-complexity Round 2) *agrees with the remedy direction*: remove or seal the generic rather than document it.
- **Recommended action:** Either seal `T` to a compiler-enforced `NppPixelType`, or remove the parameter and commit to `u8`. Do this before any `f32` path is attempted; the layout model must separate element-size from channel-count first. Aligns with the brief's "avoid speculative features."

### C12 — No test verifies pixel correctness — **Confirmed**
Even when run on a GPU, no test compares device output bytes to a reference image. `test_resize1–4` assert only layout geometry; `test_bgra_to_rgb` asserts only channel count and dimensions; the benchmarks assert nothing. So the crate's central requirement — "does it correctly resize / reorder channels?" — is unverified.
- **Confirmed by:** arch-design (Round 2 §5.2 — surfaced explicitly), arch-principal (§4.5 / Round 2 — CI never runs tests), arch-complexity and arch-security (implicit in "silently wrong output" concerns). This finding is *why* the packed-vs-pitched severity could be softened but not dismissed (CT5): there is no positive evidence the packed path is pixel-correct, only the absence of a crash.
- **Recommended action:** Add golden-image assertions for full frames **and** sub-image ROIs; make benchmarks assert correctness, not just timing.

---

## Contested Findings

Claims where architects disagreed substantially, or where a Round 2 self-correction changed a Round 1 verdict. For each, the strongest argument on each side and what would resolve it.

### CT1 — Is the `set_len` pattern *currently reachable* UB, or a dormant smell? — **Contested → resolvable by inspection; agreement on the fix**
- **For "immediate UB / most dangerous function" (arch-design R1, arch-security RISK-04 Round 1):** `set_len` on an uninitialised `Vec<u8>` declares uninitialised bytes valid; if any path reads before the copy completes, or if `chunks_mut` tiling diverges from buffer size for sub-images, it is UB and OOB device reads.
- **For "dormant, not currently reachable" (arch-principal Round 2 §2.6, arch-complexity Round 2):** trace the function — exactly `height` (or `h`) chunks are written before `from_vec` reads anything; the error path returns via `?` and `Vec<u8>` drop reads nothing; writing through `&mut [u8]` to allocated-but-uninit memory is permitted, and `u8` has no invalid bit patterns. No reachable path performs a read of uninitialised memory today. The real objection is that soundness rests on a non-local arithmetic coincidence (buffer size == chunk size × loop count) asserted nowhere, so it *becomes* UB the day stride handling changes.
- **Resolution:** Resolvable by careful inspection of the read-back loop (the panel effectively did this in Round 2 and converged on "dormant but fragile"). **Either way the action is identical** (C5): replace the pattern with zeroed/`MaybeUninit` and add a stride-correct round-trip test. The dispute is about severity labelling and remediation *priority*, not about whether to fix it.

### CT2 — "GPU memory is leaked on every allocation" — **Contested → Resolved against the claim (retracted by its author)**
- **Original claim (arch-design R1, Critical):** no `Drop` frees device memory; OOM under sustained inference load.
- **Rebuttal (arch-principal Round 2 §2.1, arch-security Round 2 §2.1, and arch-design itself Round 2 §1.1):** `rustacuda::DeviceBuffer` implements `Drop`/`cuMemFree`; `CudaImage::new` uses `DeviceBuffer::zeroed` (`image.rs:34`); there is no reference cycle (`sub_image` children hold an `Rc` to the parent; the parent holds nothing back), so the refcount reaches zero normally. The `// might wanna add a Drop` note (`image.rs:249`) is inside a *test* and concerns custom teardown *ordering*, not a missing free. **arch-design explicitly retracted this finding in Round 2 and downgraded it from Critical to Low.**
- **Resolution:** **Resolved — the leak claim does not hold.** The legitimate residual is (a) *untested* lifecycle (a test-coverage gap, not a leak) and (b) the context-before-buffer teardown-order hazard, which is re-homed as the Confirmed finding C7. This retraction removes one of the original three rejection gates.

### CT3 — Is `Persistable::save` a security vulnerability (path traversal / arbitrary write) or "merely" an API footgun? — **Contested → scoping disagreement, resolvable by establishing the trust boundary**
- **For "security risk" (arch-security RISK-05):** `PathBuf::push` of an absolute path replaces the whole path; `..` navigates upward; a caller passing `../../etc/cron.d/evil` writes outside temp. In containerised/privileged contexts this can overwrite config/executables.
- **For "not a vulnerability in this artifact" (arch-principal Round 2 §4, arch-design Round 2 §2.2, and the devil's advocate's framing):** `npp-rs` is an in-process library with no privilege boundary between caller and library; the caller already holds the process's full filesystem rights and CUDA context. "Path traversal" presumes `filename` crosses a trust boundary from an untrusted source, which is neither present in the code nor stated as an assumption. Treat it as an API-contract footgun (C9).
- **Resolution:** **Resolvable** by deciding one question: *does `filename` ever originate from an untrusted source in an intended deployment?* If yes (e.g., a service taking user-supplied output names), the security framing applies and input sanitisation is required. If no (local preprocessing utility), it is C9. The *code fix* (accept a real `&Path` / sanitise / drop the `unwrap`) satisfies both readings; only the severity label depends on the boundary decision. Missing information: an explicit threat model / intended-consumer statement (see U2).

### CT4 — Should the error type (C1) be redesigned *now*, before CUDA-11/12 support, or *after*? — **Contested → resolvable by a design spike**
- **For "fix now" (consensus C1 + arch-principal, arch-design):** the error type is breaking to change and is encoded into every operation signature; deferring multiplies rework as primitives are added.
- **For "deferring reduces total work" (devil's advocate, arch-complexity Round 2 consensus-4 challenge):** `NppStatus` is CUDA-version-specific; designing an `NppError` enum against the 10.2 status table — which the project intends to leave — risks a second breaking change when CUDA 12 lands. A version-agnostic wrapper of the raw `i32` gives little ergonomic gain unless the caller decodes it.
- **Resolution:** **Resolvable by a short design spike.** The tension dissolves if the error type wraps the raw signed `NppStatus` value (version-agnostic, non-breaking across CUDA versions) plus a small stable set of *domain* variants (NPP-error / NPP-warning / CUDA / image), deferring exhaustive per-code enumeration. That satisfies "fix the signature now" without binding to the 10.2 code table. Decision needed: confirm this hybrid is acceptable to consumers (U3).

### CT5 — Packed-vs-pitched allocation: Critical correctness/UB defect, or width-conditional performance/alignment risk? — **Contested → resolvable with an NPP documentation citation and a golden-image test**
- **For "Critical" (arch-design R3 Round 1, arch-security RISK-01-adjacent Round 1):** `CudaImage` allocates packed (stride 1920 for 640px) and feeds packed strides to NPP, while NPP's own `nppiMalloc` pitches to 2048 (`raw_tests.rs`); this "violates NPP's memory model" and risks silent miscomputation or UB.
- **For "not Critical" (devil's advocate Round 2 consensus-2 challenge; arch-design's own Round 2 §1.2 softening to Medium; arch-principal Round 2 §3.2 calibrating to Medium-High; arch-security Round 2 §2.2):** the `nStep` parameter is the *caller's declared row stride*; NPP processes whatever layout `nStep` describes and does not require `nppiMalloc` alignment for the two wrapped primitives. The packed step is internally consistent with the packed buffer. `nppiMalloc`'s pitch is a *performance* convenience, not a correctness contract. The committed `test_resize*` tests run, which is weak evidence packed strides work for the tested ops. **arch-design downgraded its own rating from Critical to Medium and removed it as a standalone rejection gate.**
- **Strongest residual concern (multiple personas):** arch-security narrows it to a *width-conditional alignment hazard* — some NPP primitives require `nStep` to be a multiple of 4/16/32; for packed RGB, widths whose `width×3` is not divisible by 4 (e.g., 427→1281, 341→1023) may return `NPP_STEP_ERROR`, which the error collapse (C1) would silently hide. arch-principal adds a definite *performance* deviation (unaligned rows defeat coalescing/texture-cache paths — undercutting the very benchmarks the crate uses to justify itself) and a *latent* risk for future stricter primitives.
- **Resolution:** **Resolvable** with two concrete artifacts the panel explicitly named: (1) an NPP documentation citation stating whether any wrapped primitive requires pitched/aligned input for *correctness* (vs. performance); (2) a golden-image test at non-4-aligned widths to detect `NPP_STEP_ERROR` or corruption. Consensus disposition: **HIGH is not supported; the defensible rating is Medium / Medium-High**, and it is **not** a standalone rejection gate. The fix is stride-alignment enforcement at construction (and carrying a real pitch in `CudaLayout`), not mandating `nppiMalloc` everywhere.

### CT6 — `in_bounds` off-by-one: HIGH memory-safety vulnerability, or benign cosmetic sloppiness? — **Contested → Resolved toward "benign," by arithmetic the panel re-derived**
- **For "HIGH / arbitrary GPU memory read-write" (arch-security RISK-01, arch-design R4, arch-principal §5 — all Round 1):** `in_bounds` uses inclusive `<=` (`image.rs:91`), admitting `x == width`; combined with `get_index` this computes an offset past the allocation, feeding raw pointer arithmetic into NPP.
- **For "benign off-by-one, not an OOB vector" (arch-principal Round 2 §1.1 + §2.2, arch-design Round 2, arch-complexity Round 2 consensus-5, arch-security Round 2 partial):** `sub_image` gates on **both** corners (`image.rs:97`); for any ROI with `w,h>0` the *far-corner* constraint `x+w ≤ width && y+h ≤ height` is the binding one — exactly the correct ROI bound. The maximum admissible far corner `(width, height)` touches byte `(height-1)*hstride + width*channels - 1 = size - 1` (in bounds). The inclusive *near*-corner check only additionally admits *degenerate* zero-area ROIs (`x=width` requires `w=0`) that never dereference. **arch-principal explicitly retracted its own Round 1 memory-safety framing**; arch-complexity walked the arithmetic and reached the same conclusion.
- **Dissent that remains (arch-security Round 2 §1.2):** arch-security still describes a *compounding* structure (corner-only validation does not check full ROI extent) and maintains the blast radius is wider than a single fencepost, though it stops short of re-asserting HIGH.
- **Resolution:** **Largely resolved toward benign.** The consensus (three personas, including two retracting their own Round 1 claims) is that no `(x,y,w,h)` passing the guard produces an OOB resize or read-back; the "attacker reads NN weights" blast radius is **not substantiated by the arithmetic**. The off-by-one is real sloppiness and should be fixed (change near-corner `<=` to `<`; add an explicit full-extent check `x+w ≤ width && y+h ≤ height`), but it is **not** a HIGH vulnerability and **not** a rejection gate. This removes the second of arch-design's original three Critical gates.

---

## Unresolved Questions

Open questions no architect could close with confidence, or that require data the debate lacked.

- **U1 — Does `rustacuda` permit safe multi-thread CUDA operations regardless of Rust ownership?** The devil's advocate (arch-complexity Round 2 consensus-1) raised a substantive, non-adversarial-sounding precondition on the C3 remedy: promoting `Rc`→`Arc<Mutex>` grants multi-thread *ownership* but not multi-thread *operational* safety, because a CUDA context under the driver API is bound to its creating thread and `rustacuda` does not expose safe context migration (`cuCtxSetCurrent`). If true, `Arc<Mutex<…>>` would hand callers a *false* safety signal while driver-level unsafety remains, and the correct fix is a per-thread/owned execution-context model (the owned-buffer + view direction, plus C8), not a lock. **What would resolve it:** confirm whether `rustacuda::DeviceBuffer` is `Send`/`Sync` and whether `rustacuda` handles context-current migration internally. This directly determines which C3 remedy is correct and should be answered *before* the C3 ADR is finalised. (Note: this is a stress-test from the designated devil's advocate; weight it as a hypothesis to verify, not as panel consensus — but it is a *checkable* hypothesis with real consequences.)

- **U2 — Is the target deployment multi-threaded, and will external consumers extend `0.0.1`?** Severity of C3 (and the scoping of CT3) hinges on this. The README says "for Neural Network processing" but does not specify threading. The devil's advocate notes the crate has no external reverse dependencies yet and that PyTorch-style data loaders often use separate *processes*, not shared in-process buffers. **What would resolve it:** an explicit statement of intended consumers and concurrency model (single-thread utility vs. multi-threaded library). If single-thread-only, C3 drops from blocker to documented limitation; if multi-threaded, it is a hard gate.

- **U3 — What error-handling granularity do real consumers need?** CT4's resolution (a raw-`NppStatus`-wrapping hybrid) assumes consumers want at least domain-level discrimination. The devil's advocate asks for a concrete consumer code path that would branch on distinct NPP codes and change behaviour usefully; if every consumer handles all errors identically (log + abort), a richer type adds little. **What would resolve it:** one or two real downstream error-handling requirements.

- **U4 — Does any wrapped NPP primitive require pitched/aligned input for correctness (not performance)?** The pivot of CT5. **What would resolve it:** an NPP documentation citation per primitive, plus golden-image tests at non-4-byte-aligned widths.

- **U5 — Does the crate actually produce correct pixels at all?** C12 establishes that *no* test asserts pixel correctness. Until golden-image tests exist, the central functional question is formally unverified — the absence of a crash is the only current evidence. This is data the debate could not access (no GPU run with reference comparison was available to the architects).

---

## Risk Register

Prioritised across all eight reports. Severities reflect **post-Round-2 consensus**, which in several cases is lower than the original Round 1 rating after self-correction. Devil's-advocate-only challenges are not treated as severity reductions unless a non-advocate persona concurred.

| # | Risk | Severity (consensus) | Flagged by | Notes |
|---|------|----------------------|------------|-------|
| 1 | `debug_assert!`-only format/stride guards stripped in `--release` → OOB device write from safe code (C2) | **High** | security, complexity, design, principal | The one genuinely reachable in-process memory hazard. Compounds with #2 and C8 into silent corruption (NEW-02). |
| 2 | Error collapse to `UnknownError`; also inverts NPP *warnings* into hard errors (C1 / NEW-01) | **High** | all four | Undiagnosable failures; warning codes (positive `NppStatus`) reported as `Err`, discarding valid results. |
| 3 | Thread-hostile `Rc<RefCell<…>>` ownership model blocks the stated NN-pipeline domain (C3) | **High** | principal, design, security, complexity(R1) | Signature-shaping. Remedy gated on U1. |
| 4 | No stream/execution-context model; correctness is emergent default-stream side-effect (C8) | **High** | principal, design, security | Latent races on op chaining; no compute/copy overlap. |
| 5 | Context-lifetime use-after-free: context drop while buffers live → `cuMemFree` on dead context (C7) | **Medium-High** | principal, design, security | The correct re-framing of the retracted "leak." Untyped invariant. |
| 6 | `Rc::clone` src/dst aliasing reaches NPP unchecked; `RefCell` gives false safety (C4 / NEW-03) | **Medium-High** | complexity, design, principal, security | Overlapping in-place resize is UB per NPP; only disjoint-by-luck in tests. |
| 7 | `set_len` over uninitialised host buffer in safe `try_from` (C5) | **Medium-High** (severity Contested CT1) | design, security, principal, complexity | "Immediate UB" vs "dormant but fragile"; fix is agreed regardless. |
| 8 | No pixel-correctness test anywhere; core function unverified (C12) | **Medium-High** | design, principal | The absence of a crash is the only evidence of correctness. |
| 9 | Generic `T` actively unsound off `u8` (folds element size into channel count) (C11) | **Medium** | complexity, design, principal | Blocks the NN-relevant `f32` path; "false promise" that mis-models. |
| 10 | Packed-vs-pitched: width-conditional `nStep` alignment risk + performance deviation (CT5) | **Medium / Medium-High** (downgraded from Critical) | design(↓), security, principal | HIGH not supported; not a rejection gate. Hidden by #2 if it errors. |
| 11 | EOL/unmaintained stack (CUDA 10.2, `rustacuda 0.1`, `bindgen 0.58`) (C10) | **Medium** | all four | Caps adoption; no CUDA 11/12 path without `build.rs` edits. |
| 12 | CI builds only, no tests/clippy/fmt, retired runner, no GPU lane; compiles against registry not local `npp-sys` (C6) | **Medium** | all four | "The one undebatable defect." No automated safety net for the riskiest code. |
| 13 | `Persistable::save` redirects to temp dir, forces `.png`, panicking `unwrap` (C9) | **Medium** (footgun) / severity Contested CT3 | complexity, security, principal, design | "Security vuln" framing depends on trust boundary (U2). |
| 14 | CI install: HTTPS but no SHA256/GPG pinning before `sudo dpkg -i` (RISK-03) | **Low-Medium** (downgraded from High) | security(↓), design, principal | "Plain HTTP" claim **retracted by its author**; TLS authenticates the CDN. CI-only, NVIDIA's own packages. |
| 15 | `npp-sys = "0.0.1"` from crates.io with path commented (RISK-06) | **Low** (contributor-workflow, not security) | complexity, security, principal, design | Typosquat/dependency-confusion framing rejected (own immutable crate; bindgen regenerates ABI). Real issue: dev bootstrap + CI tests the wrong bindings (C6). |
| 16 | Hardcoded `Device::get_device(0)`; cross-tenant timing side-channel (RISK-07) | **Low** (config limitation) | security(↓), principal, design | Side-channel framing **retracted by its author**; speculative, outside the artifact's threat model. Real residual: device not configurable, no stream (rolled into C8). |
| 17 | Inclusive `in_bounds` off-by-one (CT6) | **Low** (cosmetic; downgraded from High) | security, design, principal — **two retracted own HIGH** | Re-derived as benign; far-corner check is binding. Fix anyway (`<=`→`<` + full-extent check). |
| 18 | Half-applied trait surface: `SwapChannels` via trait but `resize` inherent; dead `CopyFromImage`/`CopyToImage` (no `self`) | **Low** | complexity, design, principal | Coherence/maintainability cost, no runtime impact. Prune or fulfil. |
| 19 | `mod raw_tests;` not `#[cfg(test)]`-gated; benchmark setup duplicated 5× | **Low** | complexity | Stylistic; inner module is gated, so zero runtime cost. |

---

## Recommended Next Steps

In priority order. Items 1–4 are signature-shaping (breaking if deferred) and should be settled by **ADR before any further NPP primitive is wrapped** — this is the panel's central, repeated instruction.

1. **Close the one reachable release-mode memory hazard first (C2).** Replace every `debug_assert!` FFI precondition with `Result`-returning validation (or `assert!`), and seal the format at the type level — newtype `CudaImageRgb8`/`NppPixelType` marker, removing or bounding the generic `T` (also resolves C11). Cheapest change with the highest safety payoff; the devil's advocate and consensus converge on the type-level seal.

2. **Decide the ownership + concurrency model via ADR (C3, C4) — but first answer U1.** Confirm whether `rustacuda` supports safe cross-thread CUDA operation. Then choose: `Arc<Mutex/RwLock<…>>` (if U1 permits) or owned-buffer `CudaImage` + borrowed `CudaImageView<'a>` (preferred if context is thread-bound). In the same change, remove the redundant `RefCell` and document/type-enforce the src/dst non-aliasing contract. Do **not** use `Rc::get_mut` (cannot work with aliasing — O1).

3. **Introduce a first-class execution-context/stream abstraction (C8, C7) via ADR.** One owned type that owns context creation, configurable device selection, stream assignment, and teardown ordering — tying `CudaImage`/`DeviceBuffer` lifetime to context lifetime in the type system. This is the common parent of the stream gap, the context use-after-free, and the hardcoded device.

4. **Design a real error type (C1), resolving CT4 with a short spike.** Wrap the raw signed `NppStatus` (version-agnostic, survives the CUDA-11/12 migration) plus a small stable set of domain variants (NPP-error / NPP-warning / CUDA / image). Fix the `status == 0` check so **positive warning codes are not treated as failures** (NEW-01).

5. **Fix the unsound/fragile host read-back (C5) and add correctness tests (C12).** Replace `Vec::with_capacity` + `set_len` with zeroed/`MaybeUninit` + verified full-write; make the read-back stride-correct for sub-images. Add **golden-image** assertions for full frames and ROIs (this is the only thing that will close U5 and validate the packed path, CT5).

6. **Repair and harden CI (C6).** Migrate off `ubuntu-18.04`; add `cargo test`, `clippy -D warnings`, `fmt --check`; stand up a GPU-capable lane (self-hosted runner or documented manual gate); repoint `npp-sys` to the workspace path so CI exercises the in-tree bindings, not the registry copy.

7. **Harden the remaining safety boundary (C9, CT6) and document invariants.** Fix the benign `in_bounds` off-by-one (`<=`→`<`) and add an explicit full-extent ROI check; fix `Persistable::save` to honour a real `&Path` (or rename) and drop the panicking `unwrap`; add `// SAFETY:` notes to every `unsafe` block stating the upheld invariant — for an FFI crate the unwritten invariants *are* the contract.

8. **Plan the CUDA 11/12 + maintained-binding migration (C10) and record it in an ADR.** Add a CUDA-version abstraction (feature flags + library-layout handling) and a `bindgen` allowlist; evaluate `cudarc` as the `rustacuda` successor. Resolve stride-alignment at construction as part of this (CT5).

9. **Lower-priority hygiene.** Add SHA256 verification before `dpkg -i` in `install_server.sh` (RISK-03 — cheap, modest value); prune or fulfil the dead trait surface (`CopyFromImage`/`CopyToImage`, the `SwapChannels`/inherent-`resize` inconsistency); `#[cfg(test)]`-gate `raw_tests` and de-duplicate benchmark setup.

**Do not** broaden NPP coverage, generalise `T`, or add pixel formats until items 1–5 are closed — every persona, including the devil's advocate, agreed that widening the surface over an unsound core multiplies the unsound paths.

---

## Appendix — Devil's Advocate Disposition (`arch-complexity`, Round 2)

For transparency, how each adversarial Round 2 challenge was weighted in the verdicts above. These were treated as stress-tests, not consensus dissent.

| Challenge | Disposition | Where reflected |
|-----------|-------------|-----------------|
| `Arc<Mutex>` does not give true thread safety; `rustacuda` context is thread-bound | **Sustained as an open question** (checkable, high-consequence) | U1; gates the C3 remedy |
| Pitched-vs-packed is performance, not correctness; NPP honours arbitrary `nStep` | **Largely sustained** — corroborated by non-advocate self-corrections; severity downgraded | CT5; converged with arch-design's own softening |
| `debug_assert!` guards check invariants the constructors already enforce | **Partially rebutted** — holds for `TryFrom` paths, fails for `CudaImage::new`'s open `ColorType` | C2 |
| `in_bounds` blast radius overstated | **Sustained** — corroborated by two personas retracting their own HIGH ratings | CT6 |
| Error-type redesign should be deferred (version-specific `NppStatus`) | **Partially sustained** — resolved via a raw-status-wrapping hybrid that fixes now without binding to 10.2 | CT4 |
| `set_len` UB is dormant in current code | **Sustained as to reachability**; fix still required | CT1, C5 |
| Alternative interpretation: "constrain scope, minimal remediation" | **Noted, not adopted as consensus** — the panel's three non-advocate dispositions remained "not production-ready"; however, the *scope-narrowing* remedies (seal `T`, fix CI) were folded into items 1 and 6 | — |

The devil's advocate's most valuable contribution was subtractive and corroborative: it pressure-tested the two Critical gates that the principal and design architects independently retracted (the leak, CT2; the bounds bug, CT6) and forced the packed-vs-pitched severity down to a defensible level (CT5) — improving the accuracy of the rejection basis without changing the panel's "not production-ready" disposition.
