# Spike: cudarc 0.9 → NPP Raw Pointer Bridge

## Overview

This document establishes the **authoritative pattern** for extracting raw device pointers from cudarc's `CudaSlice<T>` to pass to NPP's `extern "C"` functions. This pattern is the single source of truth for all NPP call sites in the crate (Steps 3, 6, 7, and 8 of Phase 2).

## Authoritative Pattern (cudarc 0.9.x)

### Allocation

```rust
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};

// alloc_zeros is SAFE in cudarc 0.9 — no `unsafe` needed.
let device: Arc<CudaDevice> = CudaDevice::new(0)?;
let buf: CudaSlice<u8> = device.alloc_zeros::<u8>(num_elements)?;
```

### Host Upload

```rust
let uploaded: CudaSlice<u8> = device.htod_sync_copy(&host_data)?;
```

### Raw Pointer Extraction (for NPP extern "C" calls)

Use the `DevicePtr` / `DevicePtrMut` traits from cudarc:

```rust
use cudarc::driver::DevicePtr as _;    // provides .device_ptr() -> CUdeviceptr (u64)
use cudarc::driver::DevicePtrMut as _; // provides .device_ptr_mut() -> CUdeviceptr
```

#### Const (read-only) pointer at element offset `img_index`:

```rust
let cu_ptr: u64 = DevicePtr::device_ptr(&buf);
let src_ptr: *const u8 = (cu_ptr + (img_index as u64) * size_of::<u8>()) as *const u8;
```

#### Mutable pointer:

```rust
let cu_ptr_mut: u64 = DevicePtrMut::device_ptr_mut(&mut buf);
let dst_ptr: *mut u8 = (cu_ptr_mut + (img_index as u64) * size_of::<u8>()) as *mut u8;
```

**IMPORTANT:** The `CudaSlice` must NOT be dropped for the duration of the NPP call. Derive the pointer and call NPP in the same statement scope.

### Borrowed Sub-Views (for CudaImageView)

cudarc's `CudaSlice` provides `.slice(range)` / `.slice_mut(range)` which return borrowed slice references. These reference types implement `DevicePtr` / `DevicePtrMut` too, so raw-pointer extraction follows the same pattern:

```rust
let sub: &CudaSlice<u8> = &buf.slice(start..end);
let sub_ptr: *const u8 = {
    let cu = DevicePtr::device_ptr(sub);
    cu as *const u8
};
```

### NPP Call (CudaSlice must outlive the raw pointer)

```rust
let status = unsafe {
    nppiResize_8u_C3R(src_ptr, nStep_bytes, src_size, src_rect,
                      dst_ptr, dst_step_bytes, dst_size, dst_rect, mode)
};
check_status(status)?;
```

### Host Read-Back (replaces old set_len pattern — C5 fixed)

```rust
let host_vec: Vec<u8> = device.dtoh_sync_copy(&buf)?;
```

### Error Type

cudarc errors are `cudarc::driver::DriverError` (NOT `CudaResultError`, which does not exist in 0.9).

## Implementation Notes

- The pattern inside the `use` + `DevicePtr::device_ptr` + `(cu_ptr + offset)` + `as *const T` sequence is structurally stable across cudarc 0.9.x.
- If the pinned cudarc 0.9.x minor version uses different names for borrowed-slice types or trait function signatures, update this file accordingly.
- Every `pub(crate) fn device_ptr(&self)` method on `CudaImageView` should cite this file.

## References

- cudarc 0.9 documentation: https://docs.rs/cudarc/0.9/cudarc/
- NPP documentation: https://docs.nvidia.com/cuda/npp/
