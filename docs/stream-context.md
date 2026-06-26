# Stream-context ordering model

This document explains the async contract of
[`StreamContext`](../npp/src/stream.rs) and how it eliminates the race
conditions identified in C8 (review finding: "no stream / execution-context
model; correctness is emergent default-stream side-effect").

## Execution model

`StreamContext` provides four execution stages with two distinct ordering
guarantees:

1. **Host-to-device copies** are performed on the CUDA default stream
   (e.g. `CudaDevice::htod_sync_copy_into`).
2. **NPP operations** are enqueued on the `StreamContext`'s forked stream
   via the `_Ctx` API.
3. **Device-side fences** ([`Fence`](../npp/src/stream.rs)) can be
   recorded between operations for cross-stream ordering
   ([`wait_for`](../npp/src/stream.rs)) or timing
   ([`elapsed_between`](../npp/src/stream.rs)).
4. **Device-to-host read-backs** (`TryFrom<&CudaImage>`) execute on the
   **NULL stream** via cudarc's `dtoh_sync_copy`.

## Ordering guarantees

### Stage 1 → Stage 2: Ordered by forked-stream creation wait

[`CudaDevice::fork_default_stream()`](https://docs.rs/cudarc/latest/cudarc/driver/struct.CudaDevice.html#method.fork_default_stream)
creates a new stream and inserts an **implicit wait** that makes all prior
default-stream work visible before the first operation on the forked stream
executes. This means **HTOD copy → NPP op** is ordered **by construction**
— no explicit synchronisation needed between upload and computation.

### Stage 2 → Stage 3: NOT ordered — requires explicit host fence

`TryFrom<&CudaImage>` performs a synchronous DtoH copy on the **NULL
stream** via `dtoh_sync_copy`. cudarc 0.9 does not expose a stream-targeted
DtoH API — the copy always goes to the NULL stream. The forked stream
(NPP ops) and the NULL stream (readback) are genuinely **unordered** unless
an explicit barrier is inserted.

The barrier is the host-blocking `synchronize()` call (via
`cuStreamSynchronize`) on the forked stream **before** the NULL-stream
copy. This is implemented in `TryFrom<&CudaImage>`:

```rust
img.ctx.synchronize()?;                    // host fence
ctx.device().dtoh_sync_copy(&img.buf)?;    // NULL-stream copy
```

### Retraction

Earlier revisions of this document stated that the readback was ordered
"by construction" on the forked stream. This was incorrect. The fence in
`TryFrom<&CudaImage>` is **load-bearing** — removing it would create a
data race between NPP work on the forked stream and the NULL-stream DtoH
copy.

## Sync points

- [`StreamContext::synchronize()`](../npp/src/stream.rs) — Host-blocking
  fence. Calls `cuStreamSynchronize` on the forked stream. Guarantees all
  prior `_Ctx` work is complete before the host continues.
- [`TryFrom<&CudaImage>`](../npp/src/image.rs) — Performs a host fence
  via `synchronize()` followed by a NULL-stream DtoH copy. This is the
  recommended readback path for single-step or chained pipelines.
- [`Fence`](../npp/src/stream.rs) — A CUDA event used for cross-stream
  ordering and device-side timing. Create via
  [`StreamContext::record_fence()`](../npp/src/stream.rs), wait on another
  stream via [`StreamContext::wait_for()`](../npp/src/stream.rs). Timing
  between two fences (same or cross-stream) uses
  [`StreamContext::elapsed_between()`](../npp/src/stream.rs).

## Why this closes C8

The original crate had **zero stream concept** — all operations used the
implicit default stream, and correctness depended on the emergent property
that `DtoH` copies block until all prior default-stream work completes.
This broke as soon as two NPP operations were chained asynchronously.

By switching to:
- **application-populated** `NppStreamContext` (not the deprecated
  `nppGetStreamContext`)
- a **dedicated forked stream** per `StreamContext`
- **explicit sync points** (`synchronize()`, `Fence` ordering, readback

every operation's ordering is deterministic and documented, not emergent.
The fence specifically addresses the cross-stream race between the forked-
stream NPP work and the NULL-stream DtoH copy.

## StreamContext is `!Send + !Sync`

`CudaStream` became `Send + Sync` in cudarc 0.19.x, but CUDA streams are
inherently thread-bound (card C7). `StreamContext` re-imposes
`!Send + !Sync` via `PhantomData<*const ()>` in its struct definition.
`Arc<StreamContext>` is intended for shared ownership within a single thread
(or under external synchronisation). Use a separate `StreamContext` per thread
if concurrent GPU work is needed.

