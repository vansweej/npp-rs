# NPP Bindings in npp-rs

> Most operation families are now **macro-generated** via the `npp-codegen` crate.
> See [docs/codegen-architecture.md](codegen-architecture.md) for the automated
> approach — this guide covers the lower-level FFI details for manual wrapping.

## The bindgen allowlist

The bindgen configuration in `npp-sys/build.rs` restricts generated bindings to
the NPP **image** domain only:

```rust
.allowlist_function("nppi.*")
.allowlist_type("Nppi.*")
.allowlist_var("Nppi.*")
```

Signal-processing symbols (`npps*`) are excluded — `wrapper.h` comments out the
`npps.h` include. This reduces bindgen output size and keeps the FFI surface
focused.

## Functions covered

Operation families are now **macro-generated** by the `npp-codegen` crate (see
[docs/codegen-architecture.md](codegen-architecture.md)). The following families
have generated impls covering the full `NppPixelType` alphabet:

| Family | Types | Channels | Source |
|--------|-------|----------|--------|
| `Resize` | `u8`, `u16`, `i16`, `f32` | C1, C3, C4 | `resize_generated.rs` |
| `SwapChannels` | `u8`, `u16`, `i16`, `f32`, `i32` | C4C3R only | `swap_channels_generated.rs` |
| `Mean` | `u8`, `u16`, `i16`, `f32` | C1, C3, C4 | `mean_generated.rs` |

## How to add a new NPP operation

1. **Ensure the symbol is in the allowlist.** If it matches `nppi.*`, `Nppi.*`,
   it's already included. If it's a signal op (`npps.*`), update `wrapper.h`
   and the allowlist.

2. **Define or extend a capability trait** in `npp/src/imageops.rs`. If the new
   operation fits an existing trait (e.g. a new resize variant), add a method
   there. If it's a new operation family, create a new trait.

3. **Implement the trait for the concrete type(s)** in a new or existing module
   under `npp/src/`. Follow the pointer-bridge pattern:

   ```rust
   // Extract raw device pointer from CudaSlice via DevicePtr trait
   let cu_ptr = cudarc::driver::DevicePtr::device_ptr(&self.buf);
   let raw_src = (*cu_ptr + self.layout.img_index as u64) as *const T;

   // For mutable pointers:
   let cu_ptr_mut = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
   let raw_dst = (*cu_ptr_mut + dst.layout.img_index as u64) as *mut T;

   // Call the NPP function
   let status = unsafe { nppiFoo(...) };
   check_status(status)?;
   ```

   See `docs/spike-cudarc-ptr-bridge.md` for the complete pattern.

4. **Handle the byte-step conversion.** NPP's `nStep` parameter is always in
   **bytes**. The layout's `height_stride` stores **element count** of `T`.
   Multiply by `size_of::<T>()` for non-`u8` types.

5. **Add the module** to `npp/src/lib.rs`.

6. **Test.** Every new FFI call needs at minimum:
   - A geometry assertion test (dimensions/stride check).
   - For correctness-critical paths: a golden-image test (see F6 in roadmap).

## Error handling

All NPP calls return `NppStatus` (`i32`). The `check_status()` function in
`error.rs` applies the `status >= 0` rule: zero and positive codes are success,
negative codes are errors. This is a deliberate fix for the original crate's
`status == 0` check which incorrectly treated positive warning codes as hard
errors (finding C1/NEW-01 in the architecture review).

## Safety invariants

- The `CudaSlice<T>` must NOT be dropped for the duration of the NPP call.
  Derive the raw pointer and make the FFI call in the same scope.
- Source and destination buffers must not overlap in memory (C4). This applies to
  **neighbourhood-gather** operations (e.g. resize samples a pixel window); aliasing
  produces undefined results. Purely **elementwise** operations may safely alias
  (see `Normalize`).
- The `Arc<CudaDevice>` must outlive all `CudaImage`s created from it (C7).

## Build system integration

The `build.rs` in `npp-sys` reads these environment variables (set by the Nix
dev shell's `shellHook`):

| Variable | Purpose |
|----------|---------|
| `CUDA_PATH` | CUDA toolkit install root |
| `NPP_LIB_PATH` | NPP shared library path (separate Nix store output) |
| `BINDGEN_EXTRA_CLANG_ARGS` | Extra clang arguments for bindgen header parsing |
| `LIBCLANG_PATH` | Path to `libclang.so` for bindgen |
