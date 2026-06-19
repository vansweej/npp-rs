# Architecture Review Board — `npp-rs`

**Artifact under review:** `npp-rs` workspace (`npp-sys` FFI bindings + `npp` safe-ish wrapper)
**Scope:** Rust bindings to NVIDIA NPP image-processing primitives for neural-network pre-processing
**Review round:** 1
**Date:** 2026-06-19
**Decision horizon:** Long-term (binding crate published to crates.io; reverse dependencies inherit its safety and ABI guarantees)

---

## 1. Summary Verdict

**REJECT** (for production) — promising foundation, not yet production-grade.

This is a competent early-stage prototype with a clean layering instinct, but it is **not acceptable for production** in its current form. The rejection is driven by a small number of *correctness and memory-safety* defects in the FFI boundary — not by missing features. The problems are concentrated and fixable; a focused remediation pass could plausibly move this to *Accept with concerns* in one iteration.

The verdict is "Reject" rather than "Accept with concerns" because the issues are **unsound** (undefined behavior reachable from safe code, memory leaks on every GPU allocation, and a host/device memory-model mismatch against NPP's documented contract). Those are not concerns to monitor in production — they are gates that must be cleared before production.

---

## 2. Problem Statement Clarity

**Rating: Weak.**

- The stated problem (README) is one paragraph: "Rust bindings to NVIDIA NPP, a subset of image processing for neural-network processing, CUDA 10.2." That communicates *intent* but is not a specification.
- **No explicit requirements** are captured anywhere in the repo: no target throughput, no supported pixel formats matrix, no error-handling contract, no safety contract for the `unsafe` FFI surface, no statement of which NPP coverage is in/out of scope.
- The crate is published at `0.0.1` and labeled `api-bindings` / `external-ffi-bindings`. That correctly signals "experimental," which is honest — but it also means the public API is already exposed to downstream users without a stability or safety contract.
- Version skew in the problem statement: README targets **CUDA 10.2** and pins an EOL toolchain (Ubuntu 18.04, `bindgen 0.58`, `rustacuda 0.1`). "11.x support will be added later" is the only roadmap signal.

**Consequence:** Without a written contract for the FFI safety invariants (alignment, lifetime, format), reviewers cannot judge "correct vs. incorrect," and downstream users cannot use the library safely by construction.

---

## 3. Architectural Coherence

**Rating: Good intent, leaky boundary.**

The layering is the strongest part of the design and is worth preserving:

```
npp-sys   →  raw bindgen FFI (unsafe, generated)            [external-ffi-bindings]
   │
npp       →  CudaImage<T> + CudaLayout (RAII-ish wrapper)
   ├── image.rs           type, conversions (image crate ⇄ GPU)
   ├── layout.rs          stride/geometry model
   ├── cuda.rs            device/context bootstrap
   ├── resize_ops.rs      nppiResize wrapper
   ├── swap_channel_ops.rs nppiSwapChannels wrapper
   └── imageops.rs        trait definitions (SwapChannels)
```

**Coherent choices:**
- Clean `-sys` / safe-wrapper split — the idiomatic Rust pattern for FFI. Good.
- `CudaLayout` mirrors the `image` crate's `SampleLayout`, giving a single geometry vocabulary and cheap interop via `From<SampleLayout>`.
- `sub_image` via `Rc<RefCell<DeviceBuffer>>` + `img_index` is an elegant zero-copy view model; passing `img_index`-offset pointers and the parent stride to NPP is the correct way to express ROIs.
- Benchmarks deliberately compare four allocation/resize paths (rust `image`, raw `cudaMalloc`, `nppiMalloc`, the wrapper). That shows real engineering diligence about the cost model.

**Incoherences that break the abstraction:**

1. **Memory model contradicts NPP's contract.** The crate's own `raw_tests.rs` proves NPP expects *pitched* allocations: `nppiMalloc_8u_C3(640, …)` returns stride **2048**, not the packed **1920**. Yet `CudaImage::new` and every `TryFrom` allocate **packed** `DeviceBuffer`s (`height_stride = width × channels`) and then pass that packed stride to `nppiResize`/`nppiSwapChannels`. It happens to run, but it discards the alignment guarantee NPP's primitives are written against. The abstraction claims to wrap NPP while quietly violating NPP's memory model. The pitched-allocation path exists *only* in benchmarks, never in the actual `CudaImage` type.

2. **Trait surface is incoherent with the impls.** `imageops.rs` declares `SwapChannels<T>` and (in `image.rs`) `CopyFromImage` / `CopyToImage`. `CopyFromImage`/`CopyToImage` are dead — never implemented, and their signatures (`copy_from_image(image: RgbImage) -> Result<(), _>` with no `self`) cannot work. `resize` is *not* behind any trait; it is an inherent method on `CudaImage<u8>`. So the "operations as traits" pattern is half-applied and inconsistent.

3. **Monomorphic despite generics.** `CudaImage<T>` is generic, but every meaningful path is hard-bound to `u8` C3/C4. `T` creates an illusion of generality the code does not deliver, and `size_of::<T>()` in `new()` interacts dangerously with the allocation bug below.

---

## 4. Alignment With Stated Requirements

**Rating: Partially aligned with intent; the few stated constraints are met.**

| Stated intent | Status |
|---|---|
| Rust bindings to NPP | Met (narrow: resize + BGRA→RGB) |
| Subset of image ops for NN pre-processing | Partially — resize is the realistic NN pre-proc primitive; coverage is otherwise minimal |
| CUDA 10.2 | Met, but pinned to an EOL stack; no 11.x/12.x |
| Linux + Windows 10 | Plausibly met in `build.rs` (per-OS link config); only Linux is exercised in CI |
| Zero-copy sub-images | Met and elegant |

Because there is no *measurable* requirement set, "alignment" can only be judged against intent. Against intent the direction is right; against an implied production bar (sound FFI, format validation, lifecycle correctness, supported-matrix clarity) it falls short.

---

## 5. Major Risks

Ordered by severity. **R1–R3 are the rejection gates.**

### R1 — Unsound `RgbImage` reconstruction (UB reachable from safe code) — *Critical*
`image.rs` `TryFrom<&CudaImage<u8>> for RgbImage`:
```rust
let mut mem_host: Vec<u8> = Vec::with_capacity(size as usize);
unsafe { mem_host.set_len(size as usize); }   // exposes uninitialized memory as initialized u8
```
`set_len` before initialization is immediate UB if any later path reads before the device copy fully writes. The subsequent per-row copy uses `width × width_stride` as the row length while host `chunks_mut` uses `width × width_stride` against a buffer sized `width × height × channels` — when source/parent strides differ from the packed assumption (i.e. any `sub_image`), the arithmetic and the `chunks_mut`/`set_len` sizes diverge, risking out-of-bounds device reads and a truncated/garbled image. This is the most dangerous single function in the crate.

### R2 — GPU memory is leaked on every allocation — *Critical*
`CudaImage` holds `Rc<RefCell<DeviceBuffer<T>>>` but there is **no `Drop`** that frees device memory, and the code comments even flag it ("might wanna add a Drop for the cuda_buf"). `rustacuda::DeviceBuffer` does free on drop, but the surrounding pattern, the `set_len`/raw-pointer juggling, and the lack of any lifecycle test mean device-memory lifetime is unverified. For a library whose entire purpose is repeated GPU pre-processing in an inference loop, an allocation-lifecycle defect is a production stopper (OOM under sustained load).

### R3 — Packed-vs-pitched memory mismatch against NPP — *Critical/Correctness*
As shown in §3(1): the type allocates packed buffers and feeds packed strides to NPP, contradicting NPP's pitched-memory contract that the repo's own tests document (stride 2048 ≠ 1920). Risk: silent miscomputation, boundary artifacts, or UB for widths/formats where NPP assumes alignment. The correct path (`nppiMalloc`) is benchmarked but never adopted by the abstraction.

### R4 — `in_bounds` off-by-one + ROI validation gaps — *High*
`in_bounds` uses `x <= ix + iw` / `y <= iy + ih` (inclusive), so a coordinate exactly at width/height passes. `sub_image` validates only the two corners via `in_bounds`, and `get_index` can compute an offset that, combined with `w`/`h`, walks past the parent buffer. ROI math feeding raw device pointers must be airtight; here it is not.

### R5 — Error model collapses to `CudaError::UnknownError` — *High*
Every NPP non-zero status and every `image` failure is mapped to `CudaError::UnknownError` (the code itself questions this: "do we really want to map an image error to a CudaError"). NPP status codes carry precise diagnostics; discarding them makes production failures undiagnosable and conflates GPU faults with host-side image errors.

### R6 — No safety documentation on a heavily-`unsafe` surface — *High*
Numerous `unsafe` blocks (raw device pointers, `as_raw_mut`, `offset`, `set_len`) carry **zero** `// SAFETY:` justification. For an FFI crate, the unwritten invariants *are* the architecture. Their absence blocks audit and makes regressions invisible.

### R7 — `RefCell<DeviceBuffer>` + `Rc` thread-affinity / aliasing hazard — *Medium*
`Rc<RefCell<…>>` makes `CudaImage` `!Send`/`!Sync` (acceptable for now) but also means two `sub_image` views of one buffer can be passed as `src` and `dst` to `resize` — and the tests *do this* (`test_resize3`: overlapping ROIs of the same parent). `borrow_mut()` is taken twice on the same `RefCell` in the same call for src and dst pointers; overlapping in-place NPP resize is undefined per NPP, and the aliasing is neither prevented nor documented.

### R8 — EOL/abandoned toolchain & supply chain — *Medium (compounding)*
`rustacuda 0.1` (unmaintained), `bindgen 0.58`, `image 0.23`, `criterion 0.3`, Ubuntu 18.04 runner, CUDA 10.2. Static-linking a long list of `*_static` NPP libs against an EOL CUDA pins consumers to a 2019 stack and an unmaintained GPU-binding ecosystem. Long-term this is a maintenance and security liability.

### R9 — CI does not validate behavior — *Medium*
`build.yml` runs `cargo build` only — no `cargo test` (and the tests need a physical GPU, which the runner lacks), no `clippy`, no `fmt`, no `cargo deny`/`audit`. So none of R1–R7 would be caught by CI. The pre-existing `.opencode/` is untracked; CI provides no behavioral gate.

---

## 6. Required Follow-ups

**Gating (must clear before any production acceptance):**

1. **Fix R1:** Replace `Vec::with_capacity` + `set_len` with zero-initialized allocation (or `MaybeUninit` + verified full-write). Make the host reconstruction stride-correct for sub-images; add a round-trip test that asserts byte-exact equality for both full images and ROIs.
2. **Fix R2:** Implement/verify deterministic device-memory release and add a lifecycle/no-leak test (allocate–drop loop under a fixed budget). Document the `Rc` sharing semantics for `sub_image`.
3. **Resolve R3:** Either allocate pitched memory (`nppiMalloc_*`) inside `CudaImage` and carry the real pitch in `CudaLayout`, **or** write an ADR explicitly justifying packed buffers with evidence that every wrapped NPP primitive is safe for packed input across the supported format/size matrix.
4. **Fix R4 & R7:** Make `in_bounds` exclusive, validate full ROI extent against the parent buffer, and forbid (or explicitly support, per NPP rules) `src`/`dst` aliasing in `resize`.
5. **Fix R5:** Introduce a dedicated error enum that preserves NPP `NppStatus` and separates GPU vs. image-codec errors.

**Required for a credible production posture (strongly recommended):**

6. Add `// SAFETY:` notes to every `unsafe` block stating the upheld invariant (R6).
7. Write a one-page contract: supported pixel formats, type params, alignment assumptions, lifetime/threading guarantees. Convert dead traits (`CopyFromImage`/`CopyToImage`) into a coherent, implemented operation trait or delete them (R3/coherence).
8. Strengthen CI (R9): `clippy -D warnings`, `fmt --check`, `cargo deny`. Provide a GPU-enabled (self-hosted/containered) lane that actually runs the test suite, or a CPU-mockable seam so logic is testable without hardware.
9. Plan the CUDA 11/12 + maintained-binding migration (R8); record the decision and timeline in an ADR.

---

## 7. Board Notes

- **Be decisive:** Reject now; the gating list is small and concrete. This is a "fix five things" reject, not a "rethink the system" reject.
- **Avoid speculative features:** Do **not** broaden format/type coverage, add more NPP ops, or generalize `T` until R1–R3 are closed. Widening the surface over an unsound core multiplies the unsound paths.
- **Long-term consequence:** This is a published binding crate; its safety contract is inherited by every reverse dependency. Shipping the current `unsafe` boundary would export UB and leaks into downstream inference pipelines. The cost of fixing the memory model now is far lower than after an ecosystem forms around it.

**Re-review trigger:** Resubmit once R1–R5 are fixed with accompanying tests and CI runs them on GPU-capable infrastructure. Expected outcome on resubmission, if met: *Accept with concerns*.
