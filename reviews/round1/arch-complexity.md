# Architectural Complexity Review — npp-rs

**Date:** 2026-06-19  
**Reviewer:** Principal Engineer (AI)  
**Codebase:** `/home/vansweej/Work/npp-rs`  
**Model:** github-copilot/claude-sonnet-4.6

---

## Summary

`npp-rs` is a Rust FFI wrapper around NVIDIA's NPP image-processing libraries. The repository follows the established Rust convention for C-library bindings (`-sys` crate + safe wrapper crate), which is correct and appropriate. The overall architecture is simple and well-scoped. However, several design decisions introduce accidental complexity that will grow as the crate matures: an interior mutability layer that is never actually used for shared mutation, hard-coded pixel formats scattered through multiple layers, a trait that declares an operation without implementing it, and a CI setup locked to a long-EOL environment. These are concrete, removable costs — not necessary ones.

---

## Findings

### 1. `Rc<RefCell<DeviceBuffer<T>>>` — Interior Mutability That Carries No Benefit

**File:** `npp/src/image.rs`, `CudaImage<T>`

```rust
pub(in crate) image_buf: Rc<RefCell<DeviceBuffer<T>>>,
```

The `Rc<RefCell<…>>` wrapper enables shared ownership with interior mutability. The only place shared ownership is used is `sub_image`, which clones the `Rc` so a sub-image aliases the parent's buffer. That is the correct idea, but `RefCell` is never used to mutate through a shared reference — both `resize` and `swap_channel_ops` receive a plain `&mut CudaImage<T>` and call `.borrow_mut()` immediately. `borrow_mut()` on a `RefCell` that has no other outstanding borrows is a runtime no-op with runtime-checked overhead and an API surface that implies shared mutation.

The actual requirement is: *two `CudaImage`s pointing to the same `DeviceBuffer`, one of which is `&mut`*. That is expressible without `RefCell` at all:

```rust
image_buf: Rc<DeviceBuffer<T>>,   // shared, immutable
```

Operations that need to write use `Rc::get_mut` (exclusive) or raw pointer offsets (already used via `as_device_ptr().offset(...)`). The `RefCell` is accidental complexity — it adds a runtime borrow-checker that the rest of the code does not actually rely on.

**Impact:** Every call to `resize` or `swap_channels` now does `borrow_mut()` twice (once for src, once for dst). The panic path of `RefCell::borrow_mut` is reachable in safe code. A new engineer reading the struct will wonder "why is this interior-mutable?" and spend time reaching a dead end.

---

### 2. `SwapChannels<T>` Trait in `imageops.rs` Is Not Used as a Trait

**File:** `npp/src/imageops.rs`

```rust
pub trait SwapChannels<T> {
    fn bgra_to_rgb(src: &CudaImage<T>, dst: &mut CudaImage<T>) -> Result<(), CudaError>;
}
```

`swap_channel_ops.rs` implements `SwapChannels<u8> for CudaImage<u8>`, and `resize_ops.rs` adds `CudaImage::resize` as an inherent method (not via a trait). Call-sites invoke `CudaImage::bgra_to_rgb(…)` — static dispatch via the trait impl, which is identical in behaviour to an inherent `impl` block.

The trait brings no benefit:
- There is only one implementation.
- The type parameter `<T>` on the trait is not used to specialise behaviour (both src and dst are already `CudaImage<u8>`).
- `resize` follows the inherent-method pattern, not the trait pattern.

The trait pattern is justified only if: (a) multiple types will implement it, or (b) call-sites use it as a bound (e.g., `fn process<I: SwapChannels<u8>>(…)`). Neither exists. Until then it adds a cognitive indirection layer: readers must find the trait, find the impl, then understand why no method is on the type directly.

**Impact:** Inconsistency — `resize` is inherent, `bgra_to_rgb` is via trait — with no architectural reason for the difference.

---

### 3. Hard-Coded `u8` and `C3` (3-Channel) Throughout the Safe Layer

**Files:** `resize_ops.rs`, `swap_channel_ops.rs`, `image.rs`

```rust
// resize_ops.rs
debug_assert!(src.layout.width_stride == 3 && dst.layout.width_stride == 3);
nppiResize_8u_C3R(…)

// swap_channel_ops.rs
debug_assert!(src.layout.width_stride == 4 && dst.layout.width_stride == 3);
nppiSwapChannels_8u_C4C3R(…)
```

`CudaImage<T>` is generic over `T`, suggesting it was designed to support `u8`, `u16`, `f32`, etc. In practice, all real operations are hard-coded to `u8` and 3-channel layout. The generic parameter currently buys nothing but false promise: `CudaImage<f32>` can be constructed but will panic or silently malfunction when passed to any operation.

The NPP library provides distinct function names for each type/channel combination (`nppiResize_8u_C3R`, `nppiResize_16u_C1R`, etc.), so generalisation would require either monomorphisation through a trait or a type-level encoding — neither of which exists here.

**This is not a demand to generalise now.** The complexity problem is the gap between the generic API surface and the narrowly-constrained implementation. That gap is a trap for callers. Either:
- Constrain `CudaImage<T>` to `T = u8` and remove the type parameter, or
- Document clearly which operations exist for which types.

**Impact:** `debug_assert!` is the only guard — in release builds a wrong-format image reaches the NPP C function with no error, producing silently wrong output or a GPU fault.

---

### 4. `Persistable` Saves to `temp_dir()` — Side Effect Buried in the Trait

**File:** `npp/src/image.rs`, `impl Persistable for CudaImage<u8>`

```rust
fn save(&self, filename: &str) -> Result<(), image::ImageError> {
    let mut tmp_dir = temp_dir();
    tmp_dir.push(filename);
    tmp_dir.set_extension("png");
    // ...
}
```

The `Persistable::save` signature takes `filename: &str`, implying the caller controls the destination. Silently redirecting to `temp_dir()` is surprising — a caller passing `"/output/result"` will write to `/tmp/result.png` without any indication. The tests call `save("resize1")` etc. and presumably rely on this temp-dir behaviour, but it is undocumented and invisible in the type system.

This is a minor but real footgun: `Persistable` is a public trait with a misleading contract.

---

### 5. `raw_tests.rs` Is a Module of Dead Test-Only Code with No Production Value

**File:** `npp/src/raw_tests.rs`

```rust
mod raw_tests;  // in lib.rs
```

The module contains only a `#[cfg(test)]` block testing NPP allocator alignment. This is useful for verifying the FFI binding is correctly set up, but it is unconditionally compiled into the library (the `mod raw_tests` is outside any `#[cfg(test)]` guard in `lib.rs`). The module body itself is guarded, but the module declaration is not.

More importantly, the module is declared in `lib.rs` as a private `mod`, which means it contributes nothing to the public API — it exists only to run tests. The conventional approach is to co-locate tests in the file they test, or to declare the module itself `#[cfg(test)]`.

---

### 6. Benchmark Boilerplate Is Repeated Five Times

**Files:** `benches/cuda_resize_image_with_imageops.rs`

Each of the five benchmark functions in `cuda_resize_image_with_imageops.rs` repeats the full CUDA init + image load + allocation setup identically (5×15 lines of identical setup). Criterion supports `BenchmarkGroup` and `bench_with_input` patterns that allow shared setup. The duplication is not a correctness issue, but it is the kind of friction that causes benchmark code to drift from the production code path over time.

---

### 7. CI Is Frozen at Ubuntu 18.04 / CUDA 10.2 (EOL)

**File:** `.github/workflows/build.yml`, `ci/install_server.sh`

```yaml
runs-on: ubuntu-18.04
```
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-npp-10-2_10.2.89-1_amd64.deb
```

Ubuntu 18.04 reached end-of-life in April 2023. The GitHub-hosted `ubuntu-18.04` runner was retired. The CI workflow likely no longer runs at all. The `install_server.sh` installs packages from a pinned URL that may be unavailable.

This means there is no working automated build validation. Any PR is accepted without build verification.

---

## Tradeoffs / Options

| Issue | Simplify Now | Leave + Document |
|-------|--------------|-----------------|
| `Rc<RefCell<…>>` | Replace with `Rc<DeviceBuffer<T>>`; use raw ptr offsets (already done) | Adds confusion about thread safety and borrow semantics |
| `SwapChannels` trait | Merge into inherent `impl`, consistent with `resize` | If other types will implement it, trait is correct — state the plan |
| Generic `<T>` with `u8`-only impls | Add a sealed marker trait or remove the type parameter | Document which types are supported |
| `Persistable` to `temp_dir` | Accept a `Path` argument; remove implicit redirect | At minimum, document and rename to `save_to_temp` |
| CI on EOL ubuntu | Update to `ubuntu-22.04` + CUDA 11/12 | Blocks all automated validation |

---

## Recommendations

Ranked by impact on ongoing complexity cost:

1. **Fix CI first.** Without a working build, every other improvement is unverifiable. Update to a supported Ubuntu + CUDA version. The `install_server.sh` approach of manually downloading `.deb` files is brittle; prefer the official CUDA apt repository or a `docker://nvidia/cuda` container.

2. **Remove `RefCell`.** Change `image_buf` to `Rc<DeviceBuffer<T>>`. The device-pointer-offset pattern already bypasses the borrow checker; `RefCell` adds overhead and confusion without adding safety. This is a one-file change.

3. **Make the type constraint honest.** Either bound `T: NppPixelType` (a sealed trait you define) so the compiler enforces supported types, or remove the type parameter entirely and commit to `u8`. The `debug_assert!` guards should become proper `Result` errors in release builds.

4. **Consolidate `bgra_to_rgb` into an inherent method.** Delete the `SwapChannels` trait or promote it to a real extension point with documented intent. The current half-use is inconsistent.

5. **Fix `Persistable::save`.** Either accept a full path, or rename and document the temp-dir behaviour explicitly. The current signature lies.

---

## Open Questions

- Is there a roadmap for supporting types other than `u8` (e.g., `f32` for neural network inference)? The answer determines whether the generic parameter is aspirational scaffolding or dead weight.
- Is `rustacuda` still the preferred CUDA abstraction? The crate has not seen activity since 2021 and does not support CUDA 11+. The upstream `cudarc` crate is the active successor.
- Are there plans to publish `npp-sys` to crates.io independently? The `npp/Cargo.toml` references the published version (`npp-sys = "0.0.1"`) with the path dependency commented out — this creates a bootstrap problem for contributors who want to develop both crates simultaneously.
