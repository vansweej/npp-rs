# Architecture Review Board — Round 2 (Peer Challenge)

**Persona:** Design reviewer (problem-statement clarity, architectural coherence, requirement alignment, risks/unknowns)
**Artifact:** `npp-rs` workspace (`npp-sys` FFI + `npp`/`npp-rs` wrapper) @ `4bdfd70`
**Round:** 2 — critique of my own Round 1 plus the three peer reports
**Date:** 2026-06-19

> **Method note.** Before writing, I re-read the source to confirm every contested claim rather than trust any report (including my own). Where a Round 1 assertion did not survive contact with the code, I say so plainly below. Citations are `file:line`.

---

## 1. Revisions to My Own Round 1 Findings

Verifying the code against the peer reports forced three corrections to my Round 1 report. I would rather retract a wrong claim than defend it.

### 1.1 RETRACT — R2 "GPU memory is leaked on every allocation (Critical)"

**This was wrong.** I claimed a leak on every allocation. The code does not leak:

- `CudaImage::new` allocates with `DeviceBuffer::zeroed(...)` (`image.rs:34`) — zero-initialised, not `Vec::with_capacity` + `set_len`. (My R1 text implied the `set_len` pattern was in the allocation path; it is not — it is *only* in the read-back `TryFrom<&CudaImage<u8>> for RgbImage` at `image.rs:161-164`.)
- `CudaImage` owns exactly one GPU resource, `Rc<RefCell<DeviceBuffer<T>>>` (`image.rs:14`). `rustacuda::DeviceBuffer` frees device memory on `Drop`, and the last surviving `Rc` clone runs that `Drop`. There is no leak under normal ownership.

The `// might wanna add a Drop` comment (`image.rs:249`) is a *test* aside, not evidence of a leak. **arch-principal's** §5 read — "device memory freeing is currently delegated entirely to rustacuda's `DeviceBuffer` drop, which is probably fine" — is the correct characterisation, and I endorse it over my own R1. The legitimate residual is *unverified lifecycle* (no allocate→drop budget test), which is a much weaker claim than "leaks on every allocation." I downgrade R2 from **Critical** to **Low (untested, likely correct)**.

### 1.2 SOFTEN — R3 "packed-vs-pitched is Critical/Correctness"

I rated the packed allocation as a critical correctness/UB defect. Having re-read the call sites, **this is overstated.** `nppiResize_8u_C3R` and `nppiSwapChannels_8u_C4C3R` take an explicit step parameter (`resize_ops.rs:78,82`; `swap_channel_ops.rs:37,39`), and the wrapper passes the packed `height_stride` consistently with its packed allocation. NPP honours an arbitrary step ≥ row-bytes; the pitched `nppiMalloc` stride (2048 vs 1920 in `raw_tests.rs:32`) is NPP's *preferred alignment for performance*, not a hard precondition for these two primitives. That none of the three peers flagged packed allocation as a correctness bug is corroborating signal that I over-weighted it.

I downgrade R3 to **Medium**, and re-scope it: the real exposure is (a) a *performance* deviation from NPP's documented allocation idiom, and (b) a *latent* alignment hazard for any future primitive that does require pitched/aligned input. It is **not** a standalone rejection gate. (It connects to a new, sharper concern — see §5.2: nothing in the test suite actually proves the packed path produces correct pixels.)

### 1.3 CORRECT THE MECHANISM + SOFTEN — R7 aliasing claim

My R7 said the `resize` path takes `borrow_mut()` "twice on the same `RefCell` in the same call" and that `test_resize3` exercises "overlapping ROIs." Both details are wrong on inspection:

- The two `borrow_mut()` calls are **sequential per-statement temporaries** (`resize_ops.rs:61-67` then `68-74`): each `RefMut` is dropped at the `;` after `as_raw()`/`as_raw_mut()` extracts a raw pointer. They never overlap, so there is **no double-borrow panic** even when `src` and `dst` share one buffer. (This is exactly why the `RefCell` provides no protection — see my endorsement of arch-complexity in §3.3.)
- `test_resize3` (`resize_ops.rs:177-178`) uses sub-images at `(1722,954,510,555)` and `(10,10,510,555)` — **disjoint** regions of one buffer, which is well-defined for NPP. So the test is *not* UB; my "overlapping ROIs" label was inaccurate.

The underlying concern survives in weaker form: nothing in the type system *prevents* a caller from passing overlapping ROIs of one buffer as `src` and `dst`, and that case is undefined per NPP and undocumented here. I keep R7 but re-rate it **Low-Medium** and fix the mechanism: the hazard is an *unenforced, undocumented* aliasing contract, not a present-in-tests defect.

### 1.4 STRENGTHEN — the `!Send`/`!Sync` posture is a present blocker, not "acceptable for now"

In R1 I wrote that `Rc`-induced `!Send`/`!Sync` was "acceptable for now." **arch-principal 4.1** changed my mind: for the stated domain (NN pre-processing on loader/worker pools), a type that cannot cross a thread boundary is a present-tense adoption wall, not a deferrable nicety. I withdraw "acceptable for now" and align with treating the ownership/concurrency model as a **High** design decision that should be settled by ADR before more operations ossify around the current type. (Endorsement in §3.1.)

### 1.5 Net effect on my verdict — still REJECT, but re-founded

My Round 1 rejection rested on R1+R2+R3. After verification, R2 is gone and R3 is no longer a gate. **The verdict does not change, but its basis is now cleaner and better-evidenced:**

1. **Release-mode unsafety** — the only guards on format/stride preconditions feeding `unsafe` NPP calls are `debug_assert!` (`resize_ops.rs:38-39`; `swap_channel_ops.rs:9-13`), which compile to nothing in `--release`. Combined with the inclusive `in_bounds` off-by-one (`image.rs:91`), a *safe* public method can drive a mis-shaped or out-of-range device pointer into NPP in a production build. (This is the peers' RISK-02/RISK-01 territory; I now treat it as my primary gate.)
2. **No correctness evidence** — see §5.2: not one test asserts pixel values, so even the happy path is unverified.
3. **Operability** — total error collapse to `UnknownError` (my R5; endorsed by two peers).

That is a tighter, more defensible REJECT than my original three gates.

---

## 2. Challenges to Peer Claims

Each challenge names the claim, gives a reason, and states what I would keep.

### 2.1 Challenge — arch-security RISK-03 ("plain HTTP download," **HIGH**)

**Claim:** CI downloads CUDA `.deb`s over "plain HTTP" with no integrity check; severity HIGH; "all published crate versions built through this pipeline must be considered potentially tainted."

**Reason it is overstated:** The premise is factually contradicted by the evidence in the same finding. Every URL in `ci/install_server.sh` is `https://` (e.g. lines 3, 31-34), and the report's own body concedes "The URL scheme is HTTPS." TLS already provides transport integrity and authenticates NVIDIA's CDN, so the headline ("plain HTTP") does not hold. The residual gap — `wget` + `dpkg -i` bypasses APT's GPG signature chain — is real but modest: it is a **CI provisioning** script for a `0.0.1` crate that publishes no signed binary artifacts and links statically at consumer build time (consumers re-link against their *own* CUDA, not CI's). Rating this HIGH on par with reachable in-process UB miscalibrates the board's risk ordering.

**What I keep:** Adding `sha256sum --check` before `dpkg -i` is sound hygiene and cheap; I endorse the remediation. I challenge only the "plain HTTP" framing and the HIGH severity — I would place this at **Low-Medium / CI hygiene**.

### 2.2 Challenge — arch-security RISK-01 / RISK-07 blast-radius framing ("attacker reads NN weights"; "timing side-channel")

**Claim:** The `in_bounds` off-by-one lets "an attacker or buggy caller read or corrupt arbitrary GPU device memory… neural network weights, model parameters" (RISK-01), and device-0 selection enables a cross-workload "timing side-channel" (RISK-07).

**Reason it is overstated:** `npp-rs` is an in-process computation library with **no privilege boundary between caller and library**. The caller already holds the process's CUDA context and can invoke NPP — or raw `cudaMemcpy` — directly; there is no less-privileged party on the other side of the API for an "attacker" to cross. The off-by-one is a genuine **soundness** defect (a safe API that can produce OOB device access = a broken safety contract), which I raised myself as R4 and fully endorse — but dressing it as a confidentiality breach imports a threat model the artifact does not have. RISK-07's cross-tenant timing side-channel is doubly speculative: it presumes a multi-tenant shared-context deployment the library neither targets nor documents, and the brief explicitly tells us to avoid speculative threats.

**What I keep:** Fix the off-by-one (RISK-01's technical core — agreed). Make device selection configurable (RISK-07's real, non-speculative kernel: hardcoding `Device::get_device(0)` at `cuda.rs:6` is a portability/flexibility limitation). I challenge the breach/side-channel narratives, not the underlying code facts.

### 2.3 Challenge (partial) — arch-complexity Finding 1 remedy ("replace with `Rc<DeviceBuffer<T>>`, use `Rc::get_mut`… a one-file change")

**Claim:** The `RefCell` is pure accidental complexity; drop to `Rc<DeviceBuffer<T>>` and write via `Rc::get_mut` or raw pointers — "a one-file change."

**Reason the remedy is wrong even though the diagnosis is right:** `Rc::get_mut` returns `None` whenever the strong count > 1 — which is precisely the `sub_image` aliasing case the type exists to serve (`image.rs:108` clones the `Rc`). So a write through any shared sub-image would fail under `get_mut`, forcing the raw-pointer bypass anyway and proving the `RefCell` was never doing load-bearing work. More importantly, "one-file change" understates the problem: removing `RefCell` without resolving the **ownership** question (arch-principal 4.1) just trades a vestigial guard for no guard. The container choice should *fall out of* the ownership/aliasing decision (owned buffer + borrowed `CudaImageView<'a>`, or `Arc<Mutex>`), not be reverse-engineered from "delete the part that looks unused."

**What I keep / endorse:** The **diagnosis** is correct and I endorse it in §3.3 — the `RefCell` confers no benefit because every borrow is released before the FFI call. I challenge only the prescription and the effort estimate.

### 2.4 Where I deliberately do **not** challenge

Per the rules, forced disagreement is not the goal. I examined these peer findings specifically for grounds to challenge and found none:

- **arch-principal 4.2 (no stream / synchronization model).** Verified: zero `Stream`/`synchronize` references; correctness rests on implicit default-stream ordering of the read-back copy. This is a real, first-class architectural gap I missed entirely in Round 1. No challenge — endorsement in §3.4.
- **arch-principal 4.3 / arch-security OBS-02 (error collapse).** Matches my R5 and the code (`resize_ops.rs:91`, `image.rs:187`). No challenge — endorsement in §3.2.
- **arch-security RISK-02 / arch-complexity 3 (`debug_assert!` stripped in release).** Verified at `resize_ops.rs:38-39`, `swap_channel_ops.rs:9-13`. This is correct and important. No challenge — endorsement in §3.5.
- **arch-complexity 7 / arch-security OBS-05 (CI on retired `ubuntu-18.04`, build-only).** Verified at `build.yml:15,19`. Correct. No challenge.

---

## 3. Endorsements

### 3.1 arch-principal 4.1 — ownership/concurrency model (`Rc<RefCell<DeviceBuffer>>` is thread-hostile)

I endorse this and let it overturn my own R1 "acceptable for now." The reason it outranks my Round 1 framing: the severity must be judged against the *stated domain*, and `Rc` (`image.rs:14`) makes `CudaImage` `!Send`/`!Sync`, so it cannot be moved to a loader thread, shared across a `rayon` pool, or held across a `tokio` await — the exact usage NN pre-processing implies. This decision is encoded into every operation signature, so its cost grows with each primitive added. It belongs in an ADR before API expansion.

### 3.2 arch-principal 4.3 + arch-security OBS-02 — error collapse destroys operability

Endorsed; this corroborates my R5. Every NPP non-zero status and every `image` failure collapses to `CudaError::UnknownError` (`resize_ops.rs:88-92`, `swap_channel_ops.rs:44-48`, `image.rs:185-188`). arch-principal's specific addition that I want to amplify: the `status == 0` success test also mis-handles NPP **positive** warning codes as failures. Preserving `NppStatus` and separating NPP/CUDA/image error domains is a pre-growth, breaking-if-deferred change.

### 3.3 arch-complexity 1 — the `RefCell` carries no benefit

Endorsed, and I can now *strengthen* it with the precise mechanism I verified: in every operation the `RefMut` from `borrow_mut()` is a per-statement temporary dropped before the raw pointer is handed to NPP (`resize_ops.rs:61-74`; `swap_channel_ops.rs:19-32`; `image.rs:170-177`). Because the borrow does not span the FFI call, the `RefCell` cannot detect or prevent the src/dst aliasing it appears to guard — it is pure ceremony plus a reachable panic path. (See my partial challenge to the *remedy* in §2.3.)

### 3.4 arch-principal 4.2 — absent stream/execution-context model

Endorsed as a gap I did not surface in Round 1. For a GPU library, the lack of any first-class stream concept means (a) no way to express compute/copy overlap, and (b) correctness depends on undocumented implicit ordering that will break the first time two ops are chained with an intermediate read. This is squarely a *design coherence* issue in my lane: the type model is missing a primitive the domain requires.

### 3.5 arch-security RISK-02 + arch-complexity 3 — `debug_assert!`-only preconditions

Endorsed, and promoted: in Round 2 this is my **primary rejection gate** (§1.5). The combination of generic `CudaImage<T>` (`image.rs:13`), `u8`-only NPP variants, and release-stripped `debug_assert!` guards means a wrong-format image reaches a C kernel with **no** runtime check in production. Replace with `Result`-returning validation or `assert!`.

### 3.6 arch-security RISK-05 + arch-complexity 4 + arch-principal §5 — `Persistable::save` contract

Endorsed as a real API-coherence defect I missed. `save(filename)` ignores the caller's path semantics, silently redirecting to `temp_dir()` and forcing `.png` (`image.rs:192-201`), with a `to_str().unwrap()` panic path. The signature lies about its contract. I weight arch-complexity's "the signature lies" framing as the design-relevant core; arch-security's path-traversal angle (`PathBuf::push` of an absolute/`..` argument) is the sharper edge of the same defect. Either accept a real `&Path` or rename to reflect the temp-dir behaviour.

---

## 4. Findings I Now Accept That No Single Report Owned

These were touched by peers; I list them because I omitted them in Round 1 and they belong on the design ledger:

- **Context lifetime is type-unchecked** (arch-principal §5; arch-security RISK-07-adjacent). `initialize_cuda_device` (`cuda.rs:4-8`) returns a `Context` the caller must keep alive; tests park it in `_ctx`. If it drops while `CudaImage`s live, device pointers dangle. Undocumented, unenforced — a coherence gap between the lifecycle the API implies and the one it guarantees.
- **`npp-sys = "0.0.1"` from crates.io, path commented** (arch-complexity Open Qs; arch-security RISK-06), at `npp/Cargo.toml:13-14`, inside a workspace whose members are `npp-sys` + `npp` (`Cargo.toml:1-5`). The published-vs-local split is a real bootstrap/coherence problem (see §5.3 for the CI implication I add).

---

## 5. New Observations (collectively hinted, none stated outright)

These emerge from combining peer threads with what the code actually does. They sit in my lane: coherence, requirement alignment, and known unknowns.

### 5.1 The generic parameter doesn't just over-promise — it is *actively* unsound off the `u8` path

arch-complexity 3 calls generic `T` a "false promise"; arch-security RISK-02 notes the stripped guards. Neither connected them to a concrete construction bug. `CudaImage::new` computes `img_size_bytes = width·height·size_of::<T>()·channels` and passes it to `DeviceBuffer::<T>::zeroed(...)` (`image.rs:32-34`), but rustacuda's `zeroed(size)` counts **elements of `T`**, not bytes — so `CudaImage::<f32>::new` over-allocates 4×, and the layout's byte-denominated `width_stride` (`layout.rs:46`) then disagrees with the element-denominated `DeviceBuffer` offsets used at `resize_ops.rs:65`. Chain it together: a `CudaImage<f32>` is constructible, mis-sized, mis-strided, and — with `debug_assert!` stripped — flows into `nppiResize_8u_C3R` in release with zero diagnostics. **Conclusion for the board:** the generic signature is not merely dead weight; it is a type signature that actively lies. Either seal `T` to a `NppPixelType` the compiler enforces, or remove the parameter and commit to `u8`. (This reinforces the brief's "avoid speculative features": the generality should be *removed*, not documented.)

### 5.2 Nothing verifies pixel correctness — the core unknown the whole review circles

Every report worries about "silently wrong output" (my R3, arch-complexity 3, arch-security RISK-02), and arch-principal notes CI never runs the tests. But the deeper gap is that **even when run on a GPU, the tests cannot detect wrong pixels.** `test_resize1/2/3/4` assert only layout geometry (`resize_ops.rs:130-133` etc.); `test_bgra_to_rgb` asserts only channel count and dimensions (`swap_channel_ops.rs:85-87`); `test_try_from_cudaimage_image` asserts only `SampleLayout` fields (`image.rs:267-272`). No test compares device output bytes to a reference image. The benchmarks (`b.iter(|| resize(...))`) assert nothing at all. **So the central requirement — "does this correctly resize / reorder channels?" — is an unverified unknown.** This is why I can soften R3's packed-vs-pitched severity but cannot dismiss it: there is no positive evidence the packed path is pixel-correct, only the absence of a crash. Any production acceptance must require golden-image assertions for full frames *and* sub-image ROIs.

### 5.3 The one thing CI does (build) may not even exercise the bindings under review

arch-principal/arch-complexity note CI only builds; RISK-06 notes the registry-vs-path split. Combine them: `build.yml:19` runs `cargo build`, and `npp` depends on `npp-sys = "0.0.1"` from the registry (`npp/Cargo.toml:14`) rather than the local member. So the sole automated check compiles `npp` against the *published* bindings, not the local `npp-sys/build.rs` output that this review is actually assessing. The local `npp-sys` is built as a workspace member but is decoupled from `npp`'s dependency edge. **Net:** CI validates neither behaviour (no tests, no GPU) nor the local FFI surface (registry dependency). The build-only CI is even weaker than "doesn't test" — it doesn't test *the code in the tree*.

### 5.4 The benchmarks compare two different memory models, so they prove less than they appear to

arch-principal observes the benchmarks measure a pessimistic serialized default-stream path. I add a coherence problem that undercuts their headline purpose (defending the crate's reason to exist): the `nppi_malloc` arm allocates **pitched** memory and copies with `cudaMemcpy2D` honouring the real stride (`cuda_resize_image_with_nppi_malloc.rs:22-44`), while the `imageops` arm uses the shipped **packed** `CudaImage` path (`cuda_resize_image_with_imageops.rs:21-27`). The two arms therefore differ in *layout*, not just in *wrapper overhead*, so any delta conflates "abstraction cost" with "packed vs pitched." And the pitched arm — the NPP-canonical idiom — is the one the abstraction does **not** ship. With §5.2 (no correctness assertion anywhere), the project's principal evidence validates neither correctness nor a like-for-like performance comparison of the production path.

---

## 6. Closing (my position only)

My Round 2 self-audit cost me one Critical finding (the leak, retracted) and softened another (packed/pitched), which I record without defensiveness — the code disproved them. The verdict is unchanged at **REJECT for production**, but better-founded: the gating issues are now (1) release-mode-reachable unsafety where `debug_assert!`-only preconditions and an inclusive `in_bounds` feed raw pointers into NPP from *safe* methods, (2) the complete absence of any pixel-correctness verification, and (3) operability collapse via `UnknownError`. The peer reports strengthened the case on two axes I under-weighted — the thread-hostile ownership model (arch-principal 4.1) and the missing stream model (arch-principal 4.2) — both of which are signature-shaping and therefore cheaper to settle by ADR now than after the API ossifies. Where peers reached past the artifact's actual threat model (the "plain HTTP" supply-chain HIGH, the "attacker reads model weights" and timing-side-channel narratives), I have challenged the framing while keeping the sound technical core. Reconciling these domains into a single ranked decision is the moderator's task, not mine.
