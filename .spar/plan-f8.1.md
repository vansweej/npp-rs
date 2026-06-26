# Feature: F8.1 — Remove vestigial `default_cuda_device()` wrapper

> **Framing (load-bearing — do not regress):** This is **dead-code cleanup**, not
> a new capability. Configurable device selection already shipped with F8 core
> (`CudaImage` is built from an `Arc<StreamContext>`; `StreamContext` requires
> an explicit ordinal via `stream_context_for(ordinal)`). `default_cuda_device()`
> is an unused convenience wrapper that is never re-exported from `lib.rs` and
> never referenced in any consumer doc. The commit and roadmap text must reflect
> this, not claim "add configurable device selection."

---

## Phase 1: Remove vestigial device-selection wrapper and reconcile docs

Commit message: refactor: remove vestigial default_cuda_device wrapper (F8.1)

### Step 1: Delete `default_cuda_device()` from `npp/src/cuda.rs` and repoint its unit test

Edit the file `npp/src/cuda.rs`.

Remove the `default_cuda_device` function entirely — its doc comment, its
`#[cfg(not(tarpaulin_include))]` attribute, and the function body. The block
to delete reads exactly:

```rust
/// Convenience wrapper: initialize device 0 (the default GPU).
#[cfg(not(tarpaulin_include))]
pub fn default_cuda_device() -> Result<Arc<CudaDevice>, NppError> {
    initialize_cuda_device(0)
}
```

In the `#[cfg(test)] mod tests` block at the bottom of the same file, the test
`test_cuda_initialize` currently calls `let result = default_cuda_device();`.
Change that single call to `let result = initialize_cuda_device(0);`. Keep the
test's `#[cfg_attr(not(feature = "gpu"), ignore)]` attribute, its `#[test]`
attribute, and its `assert!(result.is_ok());` body exactly as they are — only
the function-call identifier changes.

Do not modify the `initialize_cuda_device` function, the
`use cudarc::driver::CudaDevice;` import, or the `use std::sync::Arc;` import.
They all remain in use after this change, so no import should be removed.

### Step 2: Remove the unused `_dev` binding from the Resize ROI test

Edit the file `npp/src/resize_roi_tests.rs`.

Delete the import line:

```rust
use crate::cuda::default_cuda_device;
```

Inside the function `test_resize_roi_u8_c3_nn`, delete its first statement:

```rust
let _dev = default_cuda_device().expect("CUDA device init");
```

Do not replace the deleted statement with anything. This binding is dead code:
it allocates a second, unused CUDA device. Every allocation and the NPP call in
this test obtain their device through the `ctx` variable produced by
`stream_context_for(0)` on the next line. Leave every other line, every other
`use` statement (`CudaImage`, `ResizeInterpolation`, `stream_context_for`,
`assert_golden`, `TryFrom`), and the rest of the test body unchanged.

### Step 3: Remove the unused `_dev` binding from the SwapChannels ROI test

Edit the file `npp/src/swap_channels_roi_tests.rs`.

Delete the import line:

```rust
use crate::cuda::default_cuda_device;
```

Inside the function `test_swap_channels_roi_u8_bgra_to_rgb`, delete its first
statement:

```rust
let _dev = default_cuda_device().expect("CUDA device init");
```

Do not replace the deleted statement with anything. As in the Resize ROI test,
this binding is an unused second device; the test operates entirely through the
`ctx` variable from `stream_context_for(0)`. Leave every other line, every other
`use` statement, and the rest of the test body unchanged.

### Step 4: Repoint the status-code spike test to `initialize_cuda_device`

Edit the file `npp/tests/spike_npp_status.rs`.

Change the import line `use npp_rs::cuda::default_cuda_device;` to
`use npp_rs::cuda::initialize_cuda_device;`.

Inside the function `ensure_cuda`, change the statement
`default_cuda_device().expect("CUDA device init for spike");` to
`initialize_cuda_device(0).expect("CUDA device init for spike");`.

This file genuinely needs a bare `Arc<CudaDevice>` because it drives raw
`nppiMalloc`/`nppiResize` FFI and parks the device handle in a process-lifetime
`OnceLock`. It must call `initialize_cuda_device(0)`, not drop the call. Leave
the `#![cfg(feature = "gpu")]` inner attribute, the `OnceLock` structure, and
all other code in the file unchanged.

### Step 5: Repoint the resize-caps probe test to `initialize_cuda_device`

Edit the file `npp/tests/probe_resize_caps.rs`.

Change the import line `use npp_rs::cuda::default_cuda_device;` to
`use npp_rs::cuda::initialize_cuda_device;`.

Inside the function `probe_resize_caps`, change the statement
`let device: Arc<CudaDevice> = default_cuda_device().expect("CUDA device init");`
to
`let device: Arc<CudaDevice> = initialize_cuda_device(0).expect("CUDA device init");`.

Leave the import line `use cudarc::driver::{CudaDevice, DevicePtrMut};`
unchanged — `CudaDevice` is still required for the type annotation on the
`device` binding. Leave the `#![cfg(feature = "gpu")]` inner attribute and the
rest of the file unchanged.

### Step 6: Reconcile the F8.1 entry in `docs/roadmap.md`

Edit the file `docs/roadmap.md`.

Locate the section headed `## F8.1 — Configurable device selection`. Update it
so it reflects that F8.1 is complete and that its real deliverable was removing
a vestigial wrapper, not adding a new capability:

1. Mark the heading complete in the style of other finished entries in this file
   (change the heading to `## F8.1 — Configurable device selection *(complete)*`).
2. Rewrite the **What** and **Why** paragraphs to state: configurable device
   selection was already delivered by F8 core (the `StreamContext` pivot —
   `CudaImage` is constructed from an `Arc<StreamContext>`, and
   `stream_context_for(ordinal)` already requires an explicit ordinal); F8.1's
   actual change is the removal of the unused `default_cuda_device()` convenience
   wrapper. Remove the sentence fragment that references killing the hardcoded
   `Device::get_device(0)` — that is a `rustacuda` API that no longer exists in
   this cudarc-based codebase.
3. In the "Suggested rough sequencing" prose note near the end of the file (the
   paragraph beginning "Sequencing note:"), move F8.1 out of the list of
   remaining/independent features and into the list of completed-and-merged
   features alongside F8 (core).

Do not edit `plans/tarpaulin-coverage.md`. It references the removed function
but is an archival plan record and is intentionally left as historical context.

---

## Phase 2: Document verification commands

Commit message: docs: add verification commands for F8.1 to roadmap entry

### Step 1: Add verification commands to the F8.1 roadmap entry

Edit the file `docs/roadmap.md`.

At the end of the `## F8.1` section updated in Phase 1, add a **Verification**
paragraph listing the four commands that must pass after this change, each run
inside the Nix dev shell. Use the exact command lines:

```
nix develop . --command cargo fmt --check
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo clippy --features gpu -- -D warnings
nix develop . --command cargo test
```

Include a one-line explanation that the `--features gpu` clippy invocation is
required because the ROI test modules (`resize_roi_tests`,
`swap_channels_roi_tests`) are gated behind `#[cfg(all(test, feature = "gpu"))]`
and are otherwise never compiled, so their import hygiene would go unchecked.
Note that GPU golden *tests* remain a manual gate and must not be added to CI.
