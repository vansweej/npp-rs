# NPP Bindings in npp-rs

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

## Functions wrapped in M1

| NPP function | Wrapper | Description |
|-------------|---------|-------------|
| `nppiResize_8u_C3R` | `Resize for CudaImage<u8>` | 3-channel u8 resize |
| `nppiResize_32f_C3R` | `Resize for CudaImage<f32>` | 3-channel f32 resize (with byte-step conversion) |
| `nppiSwapChannels_8u_C4C3R` | `SwapChannels for CudaImage<u8>` | BGRA (4ch) → RGB (3ch) reorder |

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
- Source and destination buffers must be **non-overlapping** (C4). Passing
  overlapping ROIs to NPP is undefined behaviour.
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
