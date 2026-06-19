# Spar Brief — M1 re-scoping (2026-06-19)

## What happened

Socratic review of `reviews/M1-build-again-plan.md` against the actual code and
real cudarc 0.9.15 API. The original plan made assumptions about:
- `cudarc`'s ownership model (assumed similar to `rustacuda` — wrong; `CudaSlice<T>` is already `Send + Sync`)
- The set of NPP pixel types (`u8`+`f32` only — wrong; NPP has ~9 primitive types)
- Whether `image-rs`/`graphynx` integration forces type design now (resolved: no — both are boundary conversions, deferred)
- The scope fences between M1 and M2 (author overruled: C11, `image` upgrade, and full-alphabet design pulled into M1)

## Design decisions made

### Ownership model (C3/C4/C7 — largely dissolved by cudarc)
- `CudaImage<T>` owns a plain `CudaSlice<T>`. **No `Rc`, no `RefCell`, no `Arc<Mutex>`.** cudarc's `CudaSlice` is `Send + Sync`, and `.clone()` is a device-to-device copy (not aliasing).
- `sub_image` uses `CudaSlice::slice()`/`slice_mut()` → returns a **borrowed `CudaImageView<'a>`**, replacing the old `Rc::clone` aliasing. The C4 src/dst aliasing hazard moves into the borrow checker for free.
- cudarc's internal `Arc<CudaDevice>` on every `CudaSlice` prevents buffer-outlives-device (C7's residual concern).

### Type system (C11 — pulled into M1)
- `CudaImage<T: NppPixelType>` where `NppPixelType` is a marker trait covering the **full NPP primitive alphabet** (~9 types: `8u/8s/16u/16s/16f/32u/32s/32f/64f`).
- Every `T` is *constructible* and *round-trippable to `Vec<T>`* in M1.
- Operations are **capability traits** (`Resize`, `SwapChannels`, …). Impl'd per concrete type. Unsupported `(type, op)` cells have **no impl** → calling them is a **compile error**, not a runtime `NotImplemented`.
- The C11 "signature lies" defect cannot recur: a type's *existence* (constructible) is orthogonal to its *capabilities* (which ops).
- Hand-written `u8` and `f32` impls in M1 tune the trait shape for the **macro milestone** (deferred). No macro codegen in M1.
- Generic pipelines remain possible: `fn pipeline<T>(img: CudaImage<T>) where CudaImage<T>: Resize`.

### `image-rs` removed from M1 core
- Round-trip uses **`TryFrom<&CudaImage<T>> for Vec<T>`** (plain typed byte buffer) — universal for all `T: DeviceRepr`, no per-type gate, no image-rs dep.
- Image-rs conversions (`TryFrom<&CudaImage<u8>> for RgbImage`, etc.) are **deferred boundary conversions**, not core operations.
- `image` crate dependency removed from `npp/Cargo.toml` for M1. Upgrade `0.23 → 0.25+` can happen when the boundary is re-added.

### Error type
- `NppError` (thiserror-based), wrapping raw signed `NppStatus` from npp-sys.
- `status >= 0` is success (fixes C1/NEW-01 — positive NPP warning codes are warnings, not errors).
- cudarc errors wrapped as `#[from] DriverError` (**not** `CudaResultError`, which doesn't exist at this version).
- `InvalidArgument` for validation errors (forward-compat for M2's C2 hardening).
- **No `NotImplemented` variant** — made unrepresentable by capability traits.

### Crate versions
- All pinned versions are **not binding**. Use latest (cudarc `0.9`, latest bindgen, latest thiserror, etc.). The old `image 0.23.13` is replaced or removed.

## Internal M1 ordering (mandatory sequence)
1. **FFI pointer bridge spike** (bare-metal `u8` resize, no abstractions) — *must succeed before anything else proceeds.* Confirm CudaSlice → `CUdeviceptr` → offset arithmetic → NPP extern "C" call works. This is the highest-risk unknown.
2. **`u8` resize + `Vec<u8>` round-trip** — end-to-end working with hardcoded (no traits yet).
3. **Lift to `NppPixelType` + capability traits** — generalize the proven shape.
4. **Hand-write `f32` `Resize` impl** — second exemplar to validate the trait architecture.
5. **One golden-image correctness test** — a `u8` image in → resize → bytes out comparison. M1 must prove the port *computes*, not just compiles. (Addresses C12 at minimum level.)

## Plan for the `plan` agent

`reviews/M1-build-again-plan.md` is **stale on scope** but **sound on infrastructure**:
- **Phase 1** (Nix shell, `build.rs` rewrite, toolchain): keep verbatim — nothing in this brief contradicts it.
- **Phase 2** (cudarc port): **rewrite.** Steps must follow the internal ordering above. `image.rs` is fundamentally redesigned (no `Rc`/`RefCell`, no image-rs dep, `Vec<T>` round-trip, capability trait skeleton). `resize_ops.rs`/`swap_channel_ops.rs` are redesigned around capability traits with hand-written `u8` (and `f32` for resize) impls. The `build.rs` redesign from Phase 1 must also add bindgen `allowlist_function("nppi.*")` + `allowlist_type("Nppi.*")` + `allowlist_var("Nppi.*")` as planned.
- **Phase 3** (test tiering): keep verbatim — the `gpu` feature gate and `#[cfg_attr(not(feature = "gpu"), ignore)]` pattern are unaffected.
- **Phase 4** (docs): update to reflect the new design (no `image` crate core dep, `Vec<T>` round-trip, capability traits).
- **Phase 5** (README/CI): keep verbatim — README text must reference cudarc, not rustacuda, but the structure is fine.
- **Definition of Done:** update. M1 ships `u8`+`f32` resize, `Vec<T>` round-trip, the full `NppPixelType` alphabet (constructible), and capability trait architecture — but **no macro, no `image-rs` conversion, no `u16/i16/…` operation impls.**
- **OUT-OF-SCOPE:** move C11 (seal generic) *out* of the out-of-scope list (it's done in M1). Macro codegen, alphabet expansion beyond `u8`/`f32` ops, image-rs boundary, graphynx boundary, pixel-correctness test suite — all stay M2+.

## Known-wrong details in the plan's Phase 2
1. Error type is **`DriverError`**, not `CudaResultError`. (The plan guessed a non-existent type.)
2. cudarc `alloc_zeros::<u8>(n)` is **safe** (the plan labelled it `unsafe`).
3. `Vec::with_capacity` + `set_len` read-back (C5) — the plan says "preserve as-is." **Do not.** Replace with cudarc `dtoh_sync_copy` → `Vec<T>`, which deletes the C5 hazard outright.
4. `Rc<RefCell<DeviceBuffer<T>>>` → plain `CudaSlice<T>`. The plan's "if C3/C4 deferred, preserve shape" branch is dead; no mechanical port.
5. `image 0.23.13` is removed from core deps, not upgraded-in-place.

## Deferred to next milestone (not M1)
- Macro-based codegen for the remaining ~7 NPP pixel types
- `image-rs` boundary conversions (`TryFrom` for `ImageBuffer`, etc.)
- graphynx boundary conversions
- `RgbImage<u8> → RgbImage<f32>` or any `u8`/`f32` pixel conversion ops
- Golden-image test suite (C12 — M1 gets exactly one)
- C2 `debug_assert!` ⇒ `Result` hardening (still M2 as planned; C11's type seal already mitigates the worst of it)
- C8 streams/execution context
- `bgra_to_rgb` for `f32` (or confirmation there are none)
- Any `nppi*` symbol beyond what `u8` and `f32` resize need
