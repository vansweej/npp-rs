# Architectural Review — `npp-rs`

**Reviewer role:** Principal Software Architect
**Scope:** System-level architecture, not line-level code quality
**Date:** 2026-06-19
**Revision reviewed:** `main` @ `4bdfd70`

---

## 1. System Context & What This Actually Is

`npp-rs` is a **Rust FFI binding library** that exposes a subset of NVIDIA's NPP
(NVIDIA Performance Primitives) image-processing routines, targeting CUDA 10.2.
It is a *library crate published to crates.io*, not a running service. This
framing matters: the usual distributed-systems axes (horizontal scaling,
replication, request routing) do not apply. The relevant non-functional
properties for a binding crate of this kind are:

- **Safety of the FFI boundary** (the dominant concern — this is the whole reason
  the crate exists instead of callers using `npp-sys` directly)
- **Build reproducibility / portability** across CUDA versions, OSes, and toolchains
- **GPU resource lifecycle correctness** (device memory, contexts, streams)
- **Concurrency model** (can two threads use this safely? can two operations overlap?)
- **API evolvability** as more NPP primitives are wrapped
- **Operability** in the sense of: can a consumer diagnose a failure?

The architecture is a textbook two-crate split:

| Crate      | Responsibility                                             | Risk surface |
|------------|-----------------------------------------------------------|--------------|
| `npp-sys`  | Raw `bindgen`-generated bindings + link configuration     | Build, ABI   |
| `npp`      | "Safe" wrapper: `CudaImage<T>`, layout, ops               | Safety, API  |

This separation is **correct and idiomatic**. The remainder of this review
assesses how well each layer discharges its responsibility, and where the
current design will not survive contact with real workloads.

**Maturity note:** version `0.0.1`, commit history dominated by "build" /
"reorganized" commits, TODO comments left in shipped code (`image.rs:187`,
`image.rs:249`). This is **pre-alpha**. I review it against where it must get to,
not against a 1.0 bar, but the structural decisions made now are the ones that
become expensive to reverse.

---

## 2. Assumptions (made explicit, per the brief)

1. The intended consumers are Rust programs doing GPU-accelerated image
   pre-processing in ML pipelines (per README: "to be used in Neural Network
   processing"). High-throughput, possibly multi-threaded, latency-sensitive.
2. `rustacuda` 0.1 is the chosen CUDA runtime abstraction and is load-bearing.
3. The crate is expected to wrap *many more* NPP primitives over time — resize
   and channel-swap are seed examples, not the final surface.
4. Correctness of the underlying NPP kernels is NVIDIA's problem; correctness of
   *how this crate invokes them and manages memory/lifetime* is this crate's problem.
5. "11.x support will be added later" — multi-CUDA-version support is a stated goal.

If assumption (1) or (3) is wrong (e.g. this is a single-threaded personal
utility that will never grow), several findings below drop in severity. I flag
which ones.

---

## 3. Architectural Strengths

- **Clean `-sys` / safe-wrapper split.** Linking, header inclusion, and unsafe
  symbol generation are quarantined in `npp-sys`. This is the single most
  important structural decision and it is right.
- **Decoupling from NPP's stride/padding model via `CudaLayout`.** Introducing an
  explicit layout descriptor (channels, strides, `img_index`) rather than passing
  raw pointers around is a sound abstraction. It is what makes zero-copy
  sub-images expressible at all.
- **Interop with the `image` crate via `TryFrom`.** Using standard conversion
  traits as the boundary between host images and device images is the correct
  Rust idiom and keeps the surface discoverable.
- **Benchmarks exist and compare alternatives** (rust `image` crate vs. NPP via
  `nppiMalloc` vs. `cudaMalloc` vs. the safe wrapper). For a performance-oriented
  binding, baking in comparative benchmarks is a genuinely good engineering
  instinct and gives the project a way to defend its reason for existing.

---

## 4. Critical Architectural Findings

These are ordered by long-term blast radius.

### 4.1 — The concurrency model is unsound: `Rc<RefCell<DeviceBuffer>>` makes `CudaImage` thread-hostile (HIGH)

`CudaImage<T>` is defined as:

```rust
pub struct CudaImage<T> {
    image_buf: Rc<RefCell<DeviceBuffer<T>>>,
    layout: CudaLayout,
}
```

`Rc` is non-atomic and `!Send + !Sync`. Therefore **a `CudaImage` can never cross
a thread boundary**. For a library whose stated purpose is feeding neural-network
pipelines, this is a fundamental constraint, not a detail:

- GPU pre-processing is almost always done on a worker pool or a dedicated
  loader thread (cf. PyTorch `DataLoader` workers). This type cannot be moved to
  such a worker, cannot be shared across a `rayon` pool, cannot be held in a
  `tokio` task that may migrate threads.
- The choice of `Rc` appears to exist solely to support cheap sub-image aliasing
  (`sub_image` does `Rc::clone`). That is a *single-thread* optimization baked
  into the core type, and it forecloses the multi-threaded usage that the target
  domain demands.

**Trade-off being made (probably unknowingly):** cheap clone + interior
mutability on one thread, *in exchange for* total loss of parallelism across the
public type. For this domain that trade is backwards.

**Consequence / degradation behavior:** consumers will hit a compile wall the
moment they parallelize. The usual escape hatch — wrapping in the user's own
`Mutex` — does not help, because `Rc` itself is `!Send`. They will be forced to
re-architect around the crate or fork it.

**Direction (significant flaw, so a remedy is in scope):** the aliasing concern
and the ownership concern should be separated. Options, in increasing order of
effort:
- `Arc<Mutex<…>>` or `Arc<RwLock<…>>` instead of `Rc<RefCell<…>>` — restores
  `Send`/`Sync` at the cost of atomic refcounting and lock overhead per access.
- A clearer ownership model where `CudaImage` *owns* its buffer (`Send`) and
  sub-images borrow with a lifetime (`CudaImageView<'a>`), pushing aliasing into
  the type system rather than into a refcount. This is more idiomatic and avoids
  runtime locking entirely, at the cost of more involved lifetime signatures.

This decision deserves an ADR before more of the API is built on top of the
current type, because every wrapped operation signature will encode it.

### 4.2 — No stream model and no explicit synchronization: correctness rests on undocumented implicit behavior (HIGH)

Every wrapped operation (`resize`, `bgra_to_rgb`) issues its NPP call on the
**default stream** and never synchronizes explicitly. `grep` across `npp/src`
finds **zero** `synchronize` / `Stream` references.

NPP primitive calls are asynchronous with respect to the host. The code is
*currently* saved from data races only because the read-back path
(`RgbImage::try_from`) issues a `copy_to` on the default stream, which is
implicitly ordered after the kernel. This means:

- **Correctness is an emergent side effect**, not a designed contract. Nothing in
  the operation API guarantees the kernel has completed when `resize()` returns.
- There is **no way to express overlap** (compute on stream A while copying on
  stream B), which is the entire performance reason to use CUDA asynchronously.
  The benchmark numbers this crate advertises are therefore measuring a
  serialized, default-stream path — a pessimistic and non-representative mode for
  real pipelines.
- The moment anyone adds a non-synchronizing operation, or chains two ops and
  reads an intermediate, latent races appear that will manifest as
  nondeterministic pixel corruption — the worst failure class to debug.

**Architectural gap:** there is no first-class concept of a *stream* or an
*execution context* in the design. For a GPU library this is a missing primitive
on par with a web framework lacking a request object. It should be a deliberate
part of the type model (e.g. operations parameterized by, or methods on, a
stream handle), decided now, because retrofitting async/stream semantics changes
operation signatures and is therefore breaking.

### 4.3 — Total error collapse to `CudaError::UnknownError` destroys operability (HIGH)

Across `resize_ops.rs`, `swap_channel_ops.rs`, and `image.rs`, every non-zero NPP
status and every conversion failure is mapped to a single opaque value:

```rust
if status == 0 { Ok(()) } else { Err(CudaError::UnknownError) }
```

NPP returns a rich, signed `NppStatus` (negative = error, positive = warning,
with specific codes for step/size/null-pointer/interpolation violations).
Discarding it has serious consequences:

- **The library is undebuggable in production.** A consumer whose resize fails
  gets `UnknownError` whether the cause was a null pointer, a misaligned stride,
  an unsupported interpolation mode, or an out-of-memory condition. There is no
  signal to act on.
- **It conflates two different error domains.** NPP status codes are not CUDA
  driver errors; flattening NPP failures into `CudaError` is a category error
  that will mislead callers writing match arms.
- **Warnings (positive status) are silently treated as failures** by the `== 0`
  check, and conversely any future "success-with-info" path is lost.

This is a *maintainability and operability* defect more than a functional one:
the code may produce correct pixels today, but the system is opaque the moment it
does not. A dedicated error type that preserves the underlying `NppStatus` (and
keeps NPP vs. CUDA vs. image errors as distinct variants) is needed before the
surface grows, because changing the error type later is breaking.

### 4.4 — Build reproducibility and portability are fragile (MEDIUM–HIGH)

The build/link layer has several structural weaknesses that will surface as
"works on my machine" failures:

- **Hard pin to CUDA 10.2 with no version abstraction.** Static link names in
  `build.rs` (`nppc_static`, `nppial_static`, …) and the CI's CUDA 10.2 install
  assume one toolkit layout. The stated goal of CUDA 11.x support is not
  expressible without editing `build.rs`, and 11.x reorganized NPP libraries.
  There is no feature-flag or detection strategy — version support is currently a
  source edit, not a configuration.
- **Asymmetric Linux/Windows linking** (static on Linux, dynamic on Windows) means
  the two platforms have materially different runtime/deployment characteristics
  and failure modes, with no abstraction reconciling them.
- **No bindgen allowlist.** `wrapper.h` pulls in all of `npp.h` / `nppi.h` and
  generates bindings for the entire surface. This bloats compile time, couples
  the generated ABI to incidental header changes, and makes `bindgen` version
  bumps (pinned at 0.58, now quite old) higher-risk.
- **Environment-variable-driven discovery only** (`CUDA_INSTALL_DIR` /
  `CUDA_PATH`, fallback `/usr/local/cuda`). No graceful failure when CUDA is
  absent — the build simply fails inside `bindgen` with a header error rather
  than a actionable diagnostic.

**Consequence:** the crate is effectively buildable only in environments that
look exactly like the maintainer's. For a published crate this directly caps
adoption and makes downstream CI brittle. (Note: project guidance favors a Nix
dev shell for reproducibility, but **no `flake.nix` exists in this repo**, so that
reproducibility story is aspirational here.)

### 4.5 — CI does not test; it only compiles (MEDIUM)

`.github/workflows/build.yml` runs `cargo build` on `ubuntu-18.04` and stops.
There is no `cargo test`, no `clippy`, no `fmt --check`. Critically, the GitHub
runner **has no GPU**, so the entire test suite — which requires a CUDA device —
*cannot* run there and is never exercised in CI.

This is an architectural/operational gap, not a style nit:

- The safety-critical FFI invariants (stride assumptions, bounds math, layout
  conversions) are validated *only* on the maintainer's machine, if at all.
- Regressions in pointer/offset arithmetic — the highest-consequence code here —
  will not be caught before publish.
- `ubuntu-18.04` GitHub runners are **retired**; this pipeline is on borrowed time
  and will start failing for infrastructure reasons unrelated to the code.

The absence of a GPU-equipped CI path (self-hosted runner, or a documented
manual test gate) means the project has **no automated safety net** for exactly
the class of bug it is most prone to.

---

## 5. Secondary Findings (lower blast radius, noted for completeness)

- **Unsound bounds check arithmetic.** `in_bounds` uses `<=` against
  `ix + iw` / `iy + ih`, and `sub_image` checks `in_bounds(x+w, y+h)`. Off-by-one
  / overflow on `x+w` can admit out-of-range sub-images that then drive raw
  pointer offset math (`get_index`) into the NPP call. Because the result feeds
  unsafe FFI, a logic error here is a memory-safety error, not a cosmetic one.
  This sits at the architectural boundary where "safe wrapper" promises must hold.
- **`Persistable::save` hardcodes writes to the OS temp dir and `.png`**, ignoring
  the caller's `filename` path semantics, and `.unwrap()`s the conversion. A
  "save" API that silently relocates output is a surprising contract that will
  bite operators.
- **Resource lifecycle is unfinished by the author's own admission**
  (`// might wanna add a Drop` at `image.rs:249`). Device memory freeing is
  currently delegated entirely to `rustacuda`'s `DeviceBuffer` drop, which is
  probably fine, but the channel-swap/resize paths and context teardown ordering
  have not been reasoned about. Under `Rc` aliasing, *which* clone triggers the
  free, and whether the CUDA context is still alive at that point, is not
  obviously correct (see 4.1 + context lifetime below).
- **CUDA context lifetime is caller-managed and easy to get wrong.**
  `initialize_cuda_device()` returns a `Context` the caller must keep alive; tests
  bind it to `_ctx`. If a consumer lets it drop while `CudaImage`s are live, device
  pointers dangle. This invariant is entirely undocumented and unenforced by
  types. A safer design ties buffer lifetime to context lifetime.
- **`rustacuda` 0.1 is a thin, low-activity dependency.** Building the whole
  safety/concurrency model on a 0.1 crate is a supply-chain/maintenance risk worth
  an explicit decision, given it underpins 4.1 and 4.2.
- **API shape inconsistency:** operations are associated functions
  (`CudaImage::resize(&src, &mut dst, …)`) rather than methods, and the
  `SwapChannels` / `CopyFromImage` / `CopyToImage` traits are partly declared but
  unimplemented (`imageops.rs`, `image.rs:18–24`). The trait surface is
  speculative and not yet load-bearing; it should be pruned or fulfilled before it
  ossifies.

---

## 6. Failure Modes & Degradation Behavior (summary)

| Trigger                                   | Current behavior                         | Severity |
|-------------------------------------------|------------------------------------------|----------|
| Consumer parallelizes across threads      | Does not compile (`Rc` is `!Send`)       | High     |
| NPP op fails (bad stride, OOM, null)       | `UnknownError`, no diagnostic            | High     |
| Two chained ops, intermediate read         | Latent race (no explicit sync)           | High     |
| Built against CUDA 11.x                     | `build.rs` link failure                  | High     |
| Built without CUDA present                  | Opaque `bindgen` header failure          | Medium   |
| Pointer/offset regression introduced        | Not caught — CI only builds, no GPU tests| High     |
| Context dropped while images live           | Dangling device pointer (UB)             | Medium   |
| `ubuntu-18.04` runner retired               | CI breaks for infra reasons              | Medium   |

The recurring theme: **failures are silent, opaque, or deferred to the consumer's
machine.** The system has very little ability to tell its operator *why* something
went wrong, and almost no automated defense for its most dangerous code.

---

## 7. Recommendations (prioritized, architecture-level)

**Do before wrapping any more NPP primitives** (these are signature-shaping and
therefore breaking if deferred):

1. **Decide and document the ownership + concurrency model** for `CudaImage`
   (ADR). Resolve `Rc<RefCell>` → either `Arc<Mutex>` or an owned-buffer + borrowed-view
   design. This unblocks the target use case (4.1).
2. **Introduce a first-class stream / execution-context concept** and define the
   synchronization contract of every operation (4.2).
3. **Design a real error type** that preserves `NppStatus` and separates NPP /
   CUDA / image error domains (4.3).

**Do to make the crate buildable and trustworthy by others:**

4. Add a CUDA-version abstraction (feature flags + library-name/layout handling)
   and a bindgen allowlist; un-pin or update `bindgen` (4.4).
5. Stand up a **GPU-capable test path** (self-hosted runner or documented manual
   gate) and add `clippy` + `fmt` + `test` to CI; migrate off retired
   `ubuntu-18.04`. If reproducibility via Nix is intended, actually add the
   `flake.nix` (4.5).

**Do to harden the safety boundary:**

6. Fix and unit-test the bounds/offset arithmetic that feeds unsafe FFI; tie
   buffer lifetime to context lifetime; finish the `Drop`/lifecycle story (§5).

---

## 8. Bottom Line

The **macro-architecture is right**: the `-sys`/safe split, the layout
abstraction, and the `image`-crate interop are the correct skeleton for a CUDA
binding, and the benchmark discipline is commendable.

The **micro-architecture of the core type is not yet sound for its stated
purpose.** Three load-bearing decisions — the `Rc`-based ownership model, the
absent stream/synchronization model, and the collapsed error model — are each
individually sufficient to block real adoption, and all three are encoded into
the public API, meaning they get more expensive to fix with every primitive
added. Combined with a CI pipeline that compiles but never tests the
safety-critical code on a GPU, the project's greatest risk is that it will appear
to work in casual use while harboring latent concurrency and memory-safety
hazards that surface only under the parallel, high-throughput conditions its
target domain guarantees.

**Recommendation: pause API expansion and resolve §7 items 1–3 via ADRs first.**
The cost of these changes now is a few signature revisions; the cost later is a
breaking 1.0 and a fork-or-rewrite decision forced on every consumer.
