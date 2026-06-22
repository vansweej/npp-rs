# Stream-context ordering model

This document explains the async contract guaranteed by
[`StreamContext`](../npp/src/stream.rs) and how it eliminates the race
conditions identified in C8 (review finding: "no stream / execution-context
model; correctness is emergent default-stream side-effect").

## Ordered execution without explicit synchronisation

`StreamContext` guarantees ordering between these three phases **by
construction**:

1. **Host-to-device copies** — performed on the default stream (e.g.
   `CudaDevice::htod_sync_copy_into`).
2. **NPP operations** — enqueued on the `StreamContext`'s forked stream
   via the `_Ctx` API.
3. **Device-to-host read-backs** — synchronous DtoH copy on the forked
   stream (e.g. `TryFrom<&CudaImage>`).

## The guarantee

The guarantee rests on
[`CudaDevice::fork_default_stream()`](https://docs.rs/cudarc/latest/cudarc/driver/struct.CudaDevice.html#method.fork_default_stream):

> The forked stream inserts an implicit wait that makes all prior
> default-stream work visible before the first operation on the forked
> stream executes.

This means the sequence `htod_copy → NPP_op → dtoh_readback` is ordered
**without any explicit synchronisation between steps 1 and 2**. The
forked stream's creation wait is the ordering barrier.

## Sync points

When explicit synchronisation is needed (e.g. for benchmarking, or to
ensure host-side observation of device results):

- [`StreamContext::synchronize()`](../npp/src/stream.rs) — blocks the
  host until all operations enqueued on this stream complete.
- `TryFrom<&CudaImage>` read-back — performs a synchronous DtoH copy
  on the stream, which blocks until all prior work on that stream is done.

## Why this closes C8

The original crate had **zero stream concept** — all operations used the
implicit default stream, and correctness depended on the emergent property
that `DtoH` copies block until all prior default-stream work completes.
This broke as soon as two NPP operations were chained asynchronously.

By switching to:
- **application-populated** `NppStreamContext` (not the deprecated
  `nppGetStreamContext`)
- a **dedicated forked stream** per `StreamContext`
- **explicit sync points** (`synchronize()` and read-back)

every operation's ordering is deterministic and documented, not emergent.

## StreamContext is `!Send + !Sync`

`CudaStream` is `!Send + !Sync` (CUDA streams are inherently thread-bound).
`StreamContext` inherits this property. `Arc<StreamContext>` is intended
for shared ownership within a single thread (or under external
synchronisation). Use a separate `StreamContext` per thread if concurrent
GPU work is needed.

## Relation to M2

The current `StreamContext` is a standalone abstraction. Session 2 of F8
will thread `Arc<StreamContext>` onto `CudaImage` and pivot the three
existing operations (Resize, SwapChannels, Mean) to use `_Ctx` variants.
See [`docs/roadmap.md`](roadmap.md) §F8.
