# Feature: F6.2 — ROI sub-image support for Resize and SwapChannels

## Goal (restated)

Implement ROI `*_into` engines + golden tests for **Resize and SwapChannels only**,
internal-only. The view machinery (`CudaImageView`/`CudaImageViewMut`) already exists
but is dead code (`image.rs:332`, `:389`); F6.2 gives it real callers.

In a final separate docs phase, rescope the post-M1 roadmap to "idiomatic safe NPP
wrapper, no in-repo ecosystem integration" — drop F3 (image-rs), F4 (graphynx),
F10 (IPP) from this repository.

**Size signal:** this is a small feature — roughly 5 files touched for the code
(~2 engine fns + 2 thin owned wrappers, 2 new in-crate ROI golden test files,
dropping 2 `#[allow(dead_code)]` attrs, plus 1 docs phase). **Resist re-expanding it.**

## Key decisions (binding)

- **ROI scoped to Resize and SwapChannels only.** Not all five op families.
- **No public API added.** Use `pub(crate)` engines. ROI goldens are in-crate
  unit tests (`#[cfg(all(test, feature = "gpu"))]` modules in `src/`) so they
  can reach the engines. Integration tests in `npp/tests/` compile as an external
  crate and cannot reach `pub(crate)` items.
- **Default design: free `pub(crate)` `*_into` engine fns** emitted by the macros
  (no trait). A `pub(crate)` trait is acceptable only if it measurably reduces
  duplication.
- **Owned resize/bgra_to_rgb become thin wrappers** over the engine functions,
  preserving the exact same FFI call sequence so owned output stays byte-identical.
- **View readback:** `TryFrom<&CudaImageView>` uses `dtoh_sync_copy_into`
  (cudarc 0.9.15 accepts `&CudaView` as `Src: DevicePtr<T>`). The host-fence
  contract is identical to the existing owned readback: `ctx.synchronize()` then
  NULL-stream copy.
- **No `NppImageRef`/`NppImageMut` public framework** — rejected as gold-plating.

## Phases

---

## Phase 1: ROI engine for Resize

Commit message: `refactor: extract resize_into ROI engine and make owned resize a wrapper`

### Step 1.1 — Add a free engine function emitted by the resize macro

In `/home/vansweej/Work/npp-rs/npp/src/resize_macros.rs`, modify the
`impl_resize_for!` macro so that, in addition to the existing
`impl Resize for CudaImage<$rust_ty>`, it emits a free `pub(crate)` engine
function whose name is supplied by a new leading macro argument.

Change the matcher from:

```rust
($rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
```

to:

```rust
($fn_name:ident, $rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
```

Before the `impl Resize` block, emit a free function that replicates the current
owned-path FFI call sequence exactly (so output stays byte-identical):

```rust
#[allow(clippy::too_many_arguments)]
#[cfg(not(tarpaulin_include))]
pub(crate) fn $fn_name(
    src_ptr: *const $rust_ty,
    src_step_bytes: i32,
    src_w: u32,
    src_h: u32,
    src_channels: u8,
    dst_ptr: *mut $rust_ty,
    dst_step_bytes: i32,
    dst_w: u32,
    dst_h: u32,
    inter: ResizeInterpolation,
    ctx: npp_sys::NppStreamContext,
) -> Result<(), NppError> {
    if !$crate::resize_ops::mode_supported($token, inter) {
        return Err(NppError::InvalidArgument(format!(
            "Resize mode {inter:?} is not supported for type {type_token}",
            type_token = $token,
        )));
    }
    let src_size = NppiSize { width: src_w as i32, height: src_h as i32 };
    let dst_size = NppiSize { width: dst_w as i32, height: dst_h as i32 };
    let src_rect = NppiRect { x: 0, y: 0, width: src_w as i32, height: src_h as i32 };
    let dst_rect = NppiRect { x: 0, y: 0, width: dst_w as i32, height: dst_h as i32 };
    let status = unsafe {
        match src_channels {
            $(
                $ch => $sym(
                    src_ptr as *const _,
                    src_step_bytes,
                    src_size,
                    src_rect,
                    dst_ptr as *mut _,
                    dst_step_bytes,
                    dst_size,
                    dst_rect,
                    $crate::resize_ops::interpolation_mode(inter),
                    ctx,
                ),
            )+
            _ => {
                return Err(NppError::InvalidArgument(format!(
                    "unsupported channel count {} for Resize with type {}",
                    src_channels,
                    stringify!($rust_ty),
                )));
            }
        }
    };
    check_status(status)
}
```

Rewrite the existing `fn resize` body inside the owned impl to be a thin wrapper:

```rust
fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError> {
    let src_step_bytes =
        (self.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
    let dst_step_bytes =
        (dst.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
    let src_ptr =
        *cudarc::driver::DevicePtr::device_ptr(&self.buf) as *const $rust_ty;
    let dst_ptr =
        *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut $rust_ty;
    $fn_name(
        src_ptr, src_step_bytes, self.width(), self.height(), self.channels(),
        dst_ptr, dst_step_bytes, dst.width(), dst.height(),
        inter, self.ctx.raw_ctx(),
    )
}
```

Replace the macro's existing "NOTE: sub-image support … deferred to F6.2" doc
paragraph (lines ~43–45) with a note that owned `resize` is now a thin wrapper
over the `$fn_name` engine, which also serves ROI views. Keep the `# nStep unit
conversion`, `# Safety`, `# Precondition`, and `# Errors` doc sections. Preserve
the `Result<(), NppError>` return and the `check_status` contract (status >= 0
is success; positive codes are warnings, not errors).

### Step 1.2 — Update the resize macro invocations to pass engine function names

In `/home/vansweej/Work/npp-rs/npp/src/resize_generated.rs` (a committed
generated artifact; header says "GENERATED — re-run `cargo run --example
gen_resize_impls`"), prepend a unique engine-function-name argument to each
of the four `impl_resize_for!` invocations. Use `resize_into_<token>` names:

```rust
impl_resize_for!(resize_into_16s, i16, "16s", {
        1 => npp_sys::nppiResize_16s_C1R_Ctx,
        3 => npp_sys::nppiResize_16s_C3R_Ctx,
        4 => npp_sys::nppiResize_16s_C4R_Ctx,
});
impl_resize_for!(resize_into_16u, u16, "16u", {
        1 => npp_sys::nppiResize_16u_C1R_Ctx,
        3 => npp_sys::nppiResize_16u_C3R_Ctx,
        4 => npp_sys::nppiResize_16u_C4R_Ctx,
});
impl_resize_for!(resize_into_32f, f32, "32f", {
        1 => npp_sys::nppiResize_32f_C1R_Ctx,
        3 => npp_sys::nppiResize_32f_C3R_Ctx,
        4 => npp_sys::nppiResize_32f_C4R_Ctx,
});
impl_resize_for!(resize_into_8u, u8, "8u", {
        1 => npp_sys::nppiResize_8u_C1R_Ctx,
        3 => npp_sys::nppiResize_8u_C3R_Ctx,
        4 => npp_sys::nppiResize_8u_C4R_Ctx,
});
```

Keep the existing `use` statements unchanged. Do not alter the channel-to-symbol arms.

### Step 1.3 — Update the resize generator example to emit the engine name

In `/home/vansweej/Work/npp-rs/npp-codegen/examples/gen_resize_impls.rs`, update
the emission so each `impl_resize_for!(...)` invocation includes a leading
`resize_into_<token>` identifier. The emitted text must be byte-identical to the
committed `npp/src/resize_generated.rs` from Step 1.2, because a byte-identity
guard test compares generator output to the committed file. Read the existing
emission logic in this example and in `npp-codegen/src/gen_impls.rs` to match
the exact indentation, spacing, and trailing-comma formatting.

---

## Phase 2: ROI engine for SwapChannels

Commit message: `refactor: extract swap_into ROI engine and make owned bgra_to_rgb a wrapper`

### Step 2.1 — Add a free engine function emitted by the swap_channels macro

In `/home/vansweej/Work/npp-rs/npp/src/swap_channels_macros.rs`, modify
`impl_swap_channels_for!` to take a leading `$fn_name:ident` argument and emit a
free `pub(crate)` engine function in addition to the existing
`impl SwapChannels for CudaImage<$rust_ty>`.

Change the matcher from:
```rust
($rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
```
to:
```rust
($fn_name:ident, $rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
```

Emit a free function replicating the current FFI call sequence (dimension-agreement
check, one `NppiSize` from dst dims, hardcoded BGRA→RGB order `[2, 1, 0]`,
channel-dispatched call with `(src_ptr, src_step, dst_ptr, dst_step, nppi_size,
&order[0], ctx)`):

```rust
#[allow(clippy::too_many_arguments)]
#[cfg(not(tarpaulin_include))]
pub(crate) fn $fn_name(
    src_ptr: *const $rust_ty,
    src_step_bytes: i32,
    src_w: u32,
    src_h: u32,
    src_channels: u8,
    dst_ptr: *mut $rust_ty,
    dst_step_bytes: i32,
    dst_w: u32,
    dst_h: u32,
    ctx: npp_sys::NppStreamContext,
) -> Result<(), NppError> {
    if src_w != dst_w || src_h != dst_h {
        return Err(NppError::InvalidArgument(
            "src and dst dimensions must match for bgra_to_rgb".into(),
        ));
    }
    let nppi_size = NppiSize { width: dst_w as i32, height: dst_h as i32 };
    let order: [std::os::raw::c_int; 3] = [2, 1, 0];
    let status = unsafe {
        match src_channels {
            $(
                $ch => $sym(
                    src_ptr as *const _,
                    src_step_bytes,
                    dst_ptr as *mut _,
                    dst_step_bytes,
                    nppi_size,
                    &order[0],
                    ctx,
                ),
            )+
            _ => {
                return Err(NppError::InvalidArgument(format!(
                    "unsupported channel count {} for SwapChannels with type {}",
                    src_channels,
                    stringify!($rust_ty),
                )));
            }
        }
    };
    check_status(status)
}
```

Rewrite `fn bgra_to_rgb` to a thin wrapper:

```rust
fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError> {
    let src_step_bytes =
        (self.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
    let dst_step_bytes =
        (dst.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
    let src_ptr =
        *cudarc::driver::DevicePtr::device_ptr(&self.buf) as *const $rust_ty;
    let dst_ptr =
        *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut $rust_ty;
    $fn_name(
        src_ptr, src_step_bytes, self.width(), self.height(), self.channels(),
        dst_ptr, dst_step_bytes, dst.width(), dst.height(),
        self.ctx.raw_ctx(),
    )
}
```

Replace the existing "deferred to F6.2" doc paragraph (lines ~43–45) with a note
that owned `bgra_to_rgb` now wraps the `$fn_name` engine, which also serves ROI
views. Preserve the `# Safety`, `# Precondition`, `# Errors`, and `# nStep` doc
sections and the `Result<(), NppError>` / `check_status` contract.

### Step 2.2 — Update the swap_channels macro invocations to pass engine function names

In `/home/vansweej/Work/npp-rs/npp/src/swap_channels_generated.rs` (committed
generated artifact), prepend a unique engine name to each
`impl_swap_channels_for!` invocation. For the existing u8 / C4C3R invocation,
use `swap_into_8u`. Read the current invocation(s), update each to the new
signature, and keep the channel-to-symbol arms and `use` statements unchanged.

### Step 2.3 — Update the swap_channels generator example to emit the engine name

In `/home/vansweej/Work/npp-rs/npp-codegen/examples/gen_swap_channels_impls.rs`,
update the emission so each `impl_swap_channels_for!(...)` invocation includes a
leading `swap_into_<token>` identifier, byte-identical to the committed
`npp/src/swap_channels_generated.rs` from Step 2.2 (a byte-identity guard
compares them). Match the existing formatting used by the example and by
`npp-codegen/src/gen_impls.rs`.

---

## Phase 3: Strided view readback helper

Commit message: `feat: add Vec readback for CudaImageView via dtoh_sync_copy_into`

### Step 3.1 — Implement TryFrom from a borrowed view to a host Vec

In `/home/vansweej/Work/npp-rs/npp/src/image.rs`, add
`TryFrom<&CudaImageView<'_, T>> for Vec<T>`, modelled on the existing owned
`TryFrom<&CudaImage<T>> for Vec<T>` (which does `img.ctx.synchronize()?` then
`img.ctx.device().dtoh_sync_copy(&img.buf)?`).

The view path uses the same two-step host-fenced contract — host-blocking
`synchronize()` fence, then a NULL-stream device-to-host copy — but because a
view is a sub-slice (not a `CudaSlice`), use `dtoh_sync_copy_into`, which is
generic over `Src: DevicePtr<T>` and accepts a `&CudaView` (verified against
cudarc 0.9.15: `alloc.rs:339` signature; `device_ptr.rs:44` `impl DevicePtr<T>
for CudaView`). `CudaImageView<'a, T>` has `pub(crate)` fields
`ctx: Arc<StreamContext>`, `view: CudaView<'a, T>`, `layout: CudaLayout`.
Size the host buffer to the cudarc view length via
`cudarc::driver::DeviceSlice::len(&v.view)` (`dtoh_sync_copy_into` internally
asserts `src.len() == dst.len()`).

```rust
/// Copy a borrowed image view to host memory.
///
/// # Ordering contract
///
/// Identical to the owned `TryFrom<&CudaImage<T>>` path: a host-blocking
/// [`StreamContext::synchronize`] fence, then a NULL-stream DtoH copy. The
/// fence makes the NULL-stream copy safe against forked-stream NPP work.
///
/// # Strided result
///
/// The returned `Vec` spans `height * parent_height_stride` elements — for a
/// sub-image narrower than its parent, it includes inter-row parent pixels
/// between ROI rows. Callers needing a packed ROI must de-stride, or read an
/// owned destination instead.
#[cfg(not(tarpaulin_include))]
impl<'a, T: NppPixelType> TryFrom<&CudaImageView<'a, T>> for Vec<T> {
    type Error = NppError;

    fn try_from(v: &CudaImageView<'a, T>) -> Result<Self, Self::Error> {
        v.ctx.synchronize()?;
        let len = cudarc::driver::DeviceSlice::len(&v.view);
        // NppPixelType: ValidAsZeroBits + Copy, so a zeroed host buffer is a
        // valid T; the copy fully overwrites it before any read.
        let mut host: Vec<T> = vec![unsafe { std::mem::zeroed::<T>() }; len];
        v.ctx.device().dtoh_sync_copy_into(&v.view, &mut host)?;
        Ok(host)
    }
}
```

Do **not** add a `Default` bound to `T` — `NppPixelType` already bounds
`DeviceRepr + ValidAsZeroBits + Copy`, so `std::mem::zeroed::<T>()` is sound.
The `DriverError` from the copy converts to `NppError` via the existing
`#[from]` on `NppError::Cuda`.

### Step 3.2 — Keep the view device-pointer accessors as-is (leave dead_code allowing)

In `/home/vansweej/Work/npp-rs/npp/src/image.rs`, the methods
`CudaImageView::device_ptr` (≈line 379) and
`CudaImageViewMut::device_ptr_mut` (≈line 436) are annotated with both
`#[allow(dead_code)]` and `#[cfg(not(tarpaulin_include))]`.

Under this feature, the ROI tests (Phase 4) route source pointer extraction
through `CudaImageView::device_ptr()`, giving that method a real caller — but
only under the `feature = "gpu"` gate. In the default build (no `gpu`), the
caller does not exist, so the attribute must stay.

The `CudaImageViewMut::device_ptr_mut()` method remains genuinely dead: the
chosen ROI test shape (source-view → owned-dst) only needs a *read* view, never
a mutable one. **Keep both attributes unchanged.** The build must stay clean
under `cargo clippy -- -D warnings` (non-gpu).

---

## Phase 4: ROI golden tests (in-crate unit tests)

Commit message: `test: add in-crate ROI golden tests for Resize and SwapChannels`

### Step 4.1 — Add an in-crate ROI golden test module for Resize

The ROI engine `resize_into_8u` and the view accessor `device_ptr` are
`pub(crate)` and unreachable from integration tests in `npp/tests/` (those
compile as external crates). Place the ROI golden as a `#[cfg(test)]` module
**inside** `src/` so it can call them directly — adding **zero public API**.

Create `/home/vansweej/Work/npp-rs/npp/src/resize_roi_tests.rs` and wire it
into `/home/vansweej/Work/npp-rs/npp/src/lib.rs` near the existing
`#[cfg(test)] mod raw_tests;` at the end of the file:

```rust
#[cfg(all(test, feature = "gpu"))]
mod resize_roi_tests;
```

The `all(test, feature = "gpu")` gate ensures it compiles only under
`cargo test --features gpu` and never affects plain `cargo test` or non-test
builds.

**Test fixture (deterministic, no FP variance — NearestNeighbor is bit-exact):**

- Parent: 3-channel u8, width 12, height 8, per-pixel `[x*21, y*32, 128]`,
  built via `CudaImage::from_host(ctx.clone(), 3, 12, 8, &data)`.
- Source view: `parent.sub_image(0, 2, 12, 4)` — full width (contiguous source
  read), y-offset 2, height 4.
- Owned destination: `CudaImage::<u8>::new(ctx.clone(), 3, 6, 2)` — downscale
  the 12×4 ROI to 6×2.

**Route pointer extraction through the accessor methods** (refinement: gives
`device_ptr` a real caller and exercises the abstraction):

- Source const pointer via `view.device_ptr()` (returns `*const u8`).
- Source byte step via `view.layout.height_stride * std::mem::size_of::<u8>()`.
- Dst mut pointer via `*cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut u8`.
- Dst byte step via `dst.layout.height_stride * std::mem::size_of::<u8>()`.

Call `crate::resize_into_8u(...)` with source pointer/step, source
width/height/channels (from `view.width()`, `view.height()`, `view.channels()`),
dst pointer/step, dst width/height, `ResizeInterpolation::NearestNeighbor`, and
`view.ctx.raw_ctx()`.

Read the owned dst back via `Vec::try_from(&dst)` (the existing owned
`TryFrom<&CudaImage<u8>>`). Assert with
`crate::test_helpers::assert_golden(&output, EXPECTED, "resize ROI u8 C3 NN")`,
`const EXPECTED: &[u8] = &[];` left empty for GPU pinning.

Add a module-level doc comment stating:
1. This is an in-crate test **deliberately** (not in `npp/tests/`), because it
   exercises `pub(crate)` items (the ROI engine and the `device_ptr()` view
   accessor) that external tests cannot reach — **do not "fix" by moving it**.
2. The manual GPU pin procedure.
3. It reads an **owned** destination for a contiguous result, avoiding the
   strided view-readback gap.

### Step 4.2 — Add an in-crate ROI golden test module for SwapChannels

Create `/home/vansweej/Work/npp-rs/npp/src/swap_channels_roi_tests.rs` and wire
it into `lib.rs` alongside the resize ROI module:

```rust
#[cfg(all(test, feature = "gpu"))]
mod swap_channels_roi_tests;
```

**Test fixture:**

- Parent: 4-channel u8 BGRA, width 12, height 8, per-pixel
  `[x*21, y*32, 128, 255]` (`[B, G, R, A]`), via
  `CudaImage::from_host(ctx.clone(), 4, 12, 8, &data)`.
- Source view: `parent.sub_image(0, 2, 12, 4)` — full width, y-offset 2,
  height 4.
- Owned destination: `CudaImage::<u8>::new(ctx.clone(), 3, 12, 4)` —
  SwapChannels requires matching src/dst width and height (3-channel RGB).

Route source pointer extraction through `view.device_ptr()` (`*const u8`) with
source byte step `view.layout.height_stride * size_of::<u8>()`; dst mut pointer
via `*cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut u8`
with dst byte step `dst.layout.height_stride * size_of::<u8>()`.
Call `crate::swap_into_8u(...)` with source pointer/step, `view.width()` /
`view.height()` / `view.channels()`, dst pointer/step, `dst.width()` /
`dst.height()`, and `view.ctx.raw_ctx()`.
Read the owned dst back via `Vec::try_from(&dst)` and assert with
`crate::test_helpers::assert_golden(&output, EXPECTED, "swap_channels ROI u8 BGRA->RGB")`,
`const EXPECTED: &[u8] = &[];` empty for pinning.

Include the same three-point module doc comment as Step 4.1 (in-crate-on-purpose
rationale citing the `pub(crate)` engine + `device_ptr()` accessor, GPU pin
procedure, owned-dst-for-contiguous-readback note), adapted to the swap test
name.

### Step 4.3 — Confirm the in-crate test modules do not affect non-GPU or non-test builds

No new code. Verify the gating is correct:

- `cargo build` (no test, no gpu): modules excluded — no compilation, no
  effect on the `dead_code` status of the `pub(crate)` engines or view accessors.
- `cargo test` (test, no gpu): modules excluded (`feature = "gpu"` false) —
  non-gpu unit-test run unchanged; no CUDA device initialized.
- `cargo test --features gpu`: modules compiled and run.

No `#[cfg(not(tarpaulin_include))]` needed on these modules — tarpaulin runs
the non-gpu profile, under which they are excluded by the `feature = "gpu"`
gate. Confirm `cargo clippy -- -D warnings` (non-gpu) stays clean.

---

## Phase 5: Verification gate

Commit message: `chore: verify F6.2 build, lints, coverage, and unchanged owned-path goldens`

### Step 5.1 — Run the full non-GPU verification suite

No code edits. From the repository root, inside the Nix dev shell:

```bash
nix develop . --command cargo build
nix develop . --command cargo test
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo fmt --check
nix develop . --command cargo tarpaulin
```

All must pass:
- `cargo test` runs unit tests only (ROI goldens are `feature = "gpu"`-gated).
- `cargo tarpaulin` coverage must stay ≥ 90%; new engine functions and the view
  `TryFrom` are annotated `#[cfg(not(tarpaulin_include))]`, so they are excluded.

Also run the codegen byte-identity guards:

```bash
nix develop . --command cargo test -p npp-codegen
```

These compare the committed `resize_generated.rs` / `swap_channels_generated.rs`
against fresh generator output, confirming Phases 1.3 and 2.3 kept generators
and committed files in sync. If a guard fails, reconcile the generator example
with the committed `*_generated.rs` so emitted text is byte-identical.

If `cargo fmt --check` fails, run `nix develop . --command cargo fmt` (the one
permitted formatting mutation during verification) and re-check.

### Step 5.2 — Document the manual GPU pinning procedure for the ROI goldens

No code edits. Record the GPU-host-only actions required to finish F6.2. The
ROI goldens are in-crate test modules, addressed by test-name filter (not
`--test <file>`). On a machine with an NVIDIA GPU:

```bash
nix develop . --command cargo test --features gpu resize_roi_tests
nix develop . --command cargo test --features gpu swap_channels_roi_tests
```

Each first run panics "golden reference not yet pinned" and prints captured
device-output bytes. Copy each printed byte literal into the corresponding
`EXPECTED` array in `resize_roi_tests.rs` / `swap_channels_roi_tests.rs`, then
re-run to confirm the assertion passes.

Then re-run the pre-existing owned-path goldens to confirm the Phase 1–2
refactor left owned output byte-identical (the engine preserves the exact FFI
call sequence, so output *should* be identical — this is verification, not a
formality):

```bash
nix develop . --command cargo test --features gpu --test golden_resize
nix develop . --command cargo test --features gpu --test golden_swap_channels
nix develop . --command cargo test --features gpu --test golden_resize_16s
nix develop . --command cargo test --features gpu --test golden_resize_16u
nix develop . --command cargo test --features gpu --test golden_resize_32f
```

Any regression in these owned goldens means the refactor changed behaviour
beyond extracting the engine and must be investigated before F6.2 is complete.

---

## Phase 6: Documentation reconciliation

Commit message: `docs: rescope post-M1 roadmap and reflect F6.2 ROI support`

### Step 6.1 — Reflect F6.2 ROI support in the macro and module docs

Documentation only; final phase. In
`/home/vansweej/Work/npp-rs/npp/src/resize_macros.rs` and
`/home/vansweej/Work/npp-rs/npp/src/swap_channels_macros.rs`, confirm the
"deferred to F6.2" notes were replaced in Phases 1–2 with the wrapper/engine
description; update any residual "deferred to F6.2" text to state ROI sub-image
support is now implemented via the engine functions, exercised by the in-crate
ROI tests.

In the three op macros still saying "sub-image support … deferred to F6.2" —
`/home/vansweej/Work/npp-rs/npp/src/convert_macros.rs` (≈lines 59–61),
`/home/vansweej/Work/npp-rs/npp/src/mean_macros.rs` (≈lines 42–44),
`/home/vansweej/Work/npp-rs/npp/src/normalize_macros.rs` (≈lines 34–38) —
update the note to state these families remain **owned-buffer only** (ROI was
scoped to Resize and SwapChannels in F6.2; Convert/Normalize/Mean ROI is not
yet implemented). Change only doc comments in these three files, no code.

### Step 6.2 — Rescope the roadmap: drop F3, F4, F10 and add the integration-repo breadcrumb

In `/home/vansweej/Work/npp-rs/docs/roadmap.md`:

1. Mark **F6.2** (the "design complete, deferred" entry ≈line 323) as
   **implemented**: ROI `*_into` engines and in-crate golden tests for Resize
   and SwapChannels are done; the view readback helper
   (`TryFrom<&CudaImageView>`) is added; Convert/Normalize/Mean remain
   owned-only.
2. Mark **F3 — `image-rs` boundary integration** (≈line 146) and
   **F4 — `graphynx` interop** (≈line 168) as **dropped from this repository**.
   Add a breadcrumb that records the fact and prescribes no mechanism:
   ecosystem integration for the `image` and `graphynx` crates moves to
   **separate downstream integration repositories** that consume `npp-rs` as
   a published crate; any public API those repositories require from `npp-rs`
   will be decided as a deliberate future feature here when they materialize.
   Do **not** name a specific mechanism (e.g. a raw-parts constructor) — that
   is an unmade design decision.
3. Mark **F10 — IPP bindings** (≈line 444) as **out of scope / separate
   project** — a distinct library (Intel IPP, CPU) belonging to its own
   repository.
4. Update the "Suggested rough sequencing" diagram (≈lines 461–472) and the
   closing "Sequencing note" (≈lines 474–479) to remove F3, F4, and F10,
   leaving the remaining open features (F5, F5.3, F6.1, F7, F8.1, F8.2, F9) as
   the post-F6.2 menu.

Preserve the "Resolved decisions (binding)" table (lines 19–28) unchanged — in
particular the `roadmap.md:27` "Safe, idiomatic Rust; rewrite rather than
faithfully port" row remains the governing philosophy.

### Step 6.3 — Reconcile AGENTS.md scope statements

In `/home/vansweej/Work/npp-rs/AGENTS.md`, review the "Current state vs.
target" section and feature-status lines for references to F3/F4/F10 or
ecosystem integration as in-scope future work. Update them to reflect that
`image`/`graphynx` integration is now downstream (separate repos) and IPP is
a separate project, consistent with the roadmap. Confirm the C12 golden-tests
line and the test-tier description remain accurate (full-frame goldens plus the
new in-crate ROI goldens for Resize and SwapChannels). Documentation edits only;
do not alter build/test command instructions.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Strided-readback trap (view `len` is `h * parent_height_stride`, cudarc asserts src/dst len equality) | High if reading a dst view directly | Wrong golden bytes | Sidestepped: the ROI test reads an **owned** dst (contiguous). The view-readback helper documents the strided gap. |
| R2 | Owned-path regression during macro refactor | Low (engine preserves exact call sequence) | Silent wrong pixels | Phase 5.2 reruns ALL existing GPU goldens — owned output must be byte-identical. |
| R3 | cudarc `dtoh_sync_copy_into` API doesn't accept `&CudaView` on pinned version | **RESOLVED** — cudarc 0.9.15 confirmed (`alloc.rs:339`; `device_ptr.rs:44`) | — | Phase 0 was run: signature exists, `DevicePtr for CudaView` impl exists. |
| R4 | Codegen byte-identity guards fail because generator and committed file disagree on engine-name formatting | Medium if Step 1.3/2.3 formatting doesn't match | Phase 5.1 blocks | Fix the generator emission to match the committed `*_generated.rs` files exactly. |

## Dependencies

- **Phase 0 (already resolved):** cudarc 0.9.15 `dtoh_sync_copy_into` API confirmed.
- **Phase 1–2:** None — self-contained macro refactors.
- **Phase 3:** cudarc 0.9.15 (confirmed).
- **Phase 4:** Phases 1–3 complete (engines + readback exist).
- **Phase 5:** Phases 1–4 complete.
- **Phase 6:** All previous phases (docs reflect the shipped state).

---

## Notes for the executor

1. **Compile after every macro edit** — macro syntax errors are caught at use-site
   (the `*_generated.rs` files), so a broken macro won't surface until the
   invocations compile. Compile with `cargo build -p npp-rs` after each macro step.
2. **`resize_into_8u` and `swap_into_8u` live at the crate root** — the macros
   are `#[macro_export]` and the `*_generated.rs` files are root modules
   (`lib.rs` declares `pub mod resize_generated;` etc.), so the free fns are at
   `crate::resize_into_8u`, `crate::swap_into_8u`. Use these paths in Phase 4
   when calling them.
3. **Owned `CudaImage` has no `device_ptr_mut()` method** — only the *views*
   have it. The owned-dst mut pointer must go through the cudarc trait:
   `*cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut u8`.
   This is what every current macro already does.
4. **Staged pinning convention:** the ROI goldens follow the same
   inert-until-pinned discipline as all existing goldens (`test_helpers.rs:21-27`).
   Non-GPU builders complete Phases 1–3, 5.1, and 6, leaving only the GPU pin
   pass (Phase 5.2) for a machine with a device.
5. **`CudaImageViewMut::device_ptr_mut` stays dead** — the chosen ROI test shape
   (source-view → owned-dst) never needs a mutable view. This is accepted as a
   consequence; it gets its first caller when a future feature reads back a
   mutable ROI view directly.
