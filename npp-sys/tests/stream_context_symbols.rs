//! Compile-time arity check for `_Ctx` symbols and `NppStreamContext` type.
//!
//! Each test coerces a known `_Ctx` symbol to a fully-typed function pointer.
//! If the symbol is missing or has the wrong arity/parameter types, the
//! coercion fails at compile time. This is the strongest non-GPU check
//! available without calling the FFI.
//!
//! # Phase 0 findings (recorded 2026-06-22)
//!
//! All four expected `_Ctx` symbols exist with trailing `NppStreamContext`
//! passed **by value** (not by pointer).
//!
//! | Symbol | Status | Base params | Total params |
//! |--------|--------|-------------|--------------|
//! | `nppiResize_8u_C1R_Ctx` | ✅ exists | 9 | 10 |
//! | `nppiSwapChannels_8u_C4C3R_Ctx` | ✅ exists | 6 | 7 |
//! | `nppiMean_8u_C1R_Ctx` | ✅ exists | 5 | 6 |
//! | `nppiMeanGetBufferHostSize_8u_C1R_Ctx` | ✅ exists | 2 | 3 |
//!
//! **Mean buffer-size out-pointer type:** `*mut usize` (NPP `size_t*`)
//! — confirmed on both `_Ctx` and non-`_Ctx` variants.
//!
//! **`NppStreamContext` derives:** `Debug, Copy, Clone` — `raw_ctx()` returns
//! by value.
//!
//! **`cudaStream_t` definition:** `*mut CUstream_st` (opaque pointer,
//! cross-crate cast from `cudarc::driver_sys::CUstream` is valid).
//!
//! **Struct size:** 48 bytes (field `nReserved0: c_int` only).

use npp_sys::*;
use std::os::raw::c_int;

/// Resize: SRC+STEP, SRC_SIZE, SRC_RECT, DST+STEP, DST_SIZE, DST_RECT, INTERP
/// Base: 9 params. _Ctx: +1 (trailing NppStreamContext by value) = 10.
#[test]
fn resize_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        *const Npp8u,
        c_int,
        NppiSize,
        NppiRect,
        *mut Npp8u,
        c_int,
        NppiSize,
        NppiRect,
        c_int,
        NppStreamContext,
    ) -> NppStatus = nppiResize_8u_C1R_Ctx;
}

/// SwapChannels C4C3R: SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER
/// Base: 6 params. _Ctx: +1 = 7.
#[test]
fn swap_channels_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        *const Npp8u,
        c_int,
        *mut Npp8u,
        c_int,
        NppiSize,
        *const c_int,
        NppStreamContext,
    ) -> NppStatus = nppiSwapChannels_8u_C4C3R_Ctx;
}

/// Mean: SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR
/// Base: 5 params. _Ctx: +1 = 6.
#[test]
fn mean_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        *const Npp8u,
        c_int,
        NppiSize,
        *mut Npp8u,
        *mut Npp64f,
        NppStreamContext,
    ) -> NppStatus = nppiMean_8u_C1R_Ctx;
}

/// MeanGetBufferHostSize: SIZE, OUT_SCALAR (buffer size pointer)
/// Base: 2 params. _Ctx: +1 = 3.
/// hpBufferSize is *mut usize (NPP size_t*), confirmed Phase 0.3a.
#[test]
fn mean_buffer_size_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(NppiSize, *mut usize, NppStreamContext) -> NppStatus =
        nppiMeanGetBufferHostSize_8u_C1R_Ctx;
}

/// Verify NppStreamContext is accessible, has the expected size and derives
/// Copy/Clone (so raw_ctx() can return by value).
#[test]
fn npp_stream_context_type_properties() {
    // Size: 48 bytes (nReserved0: c_int, not [0i64; 16])
    assert_eq!(
        std::mem::size_of::<NppStreamContext>(),
        48usize,
        "NppStreamContext size changed; verify struct layout against nppdefs.h"
    );

    // Alignment: 8 bytes (due to nSharedMemPerBlock: usize)
    assert_eq!(
        std::mem::align_of::<NppStreamContext>(),
        8usize,
        "NppStreamContext alignment changed"
    );

    // Compile-time check: Copy+Clone are derived (fails if not)
    fn assert_is_copy<T: Copy>() {}
    fn assert_is_clone<T: Clone>() {}
    assert_is_copy::<NppStreamContext>();
    assert_is_clone::<NppStreamContext>();
}

/// Verify cudaStream_t is an opaque pointer type.
#[test]
fn cuda_stream_t_is_opaque_pointer() {
    // Compile-time: must be *mut to some type
    fn assert_is_mut_ptr<T>(_: *mut T) {}
    let ptr: cudaStream_t = std::ptr::null_mut();
    assert_is_mut_ptr(ptr);

    // Size must match pointer size
    assert_eq!(
        std::mem::size_of::<cudaStream_t>(),
        std::mem::size_of::<*mut std::ffi::c_void>(),
        "cudaStream_t must be pointer-sized"
    );
}
