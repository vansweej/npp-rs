# Getting Started with npp-rs

## Prerequisites

- **Nix** with flakes enabled (see [nixos.wiki](https://nixos.wiki/wiki/Flakes))
- **NVIDIA GPU** with the proprietary driver installed
- This repository cloned

## Build

```bash
# Enter the Nix dev shell (provides CUDA toolkit, NPP, libclang, Rust toolchain)
nix develop .

# Build all crates
cargo build

# Build just the FFI crate (useful for debugging bindgen)
cargo build -p npp-sys

# Build the safe wrapper crate
cargo build -p npp-rs
```

## Test Tiers

### Unit tests (no GPU required)

```bash
cargo test --no-default-features
```

These tests cover pure-logic items: layout computations, error handling, etc.
They do not initialize a CUDA device or allocate GPU memory.

### GPU-dependent tests (requires hardware)

```bash
cargo test --features gpu
```

These tests require an NVIDIA GPU with the proprietary driver. CI has no GPU lane;
this is a documented manual gate for GPU hosts.

### Golden-image test

A single correctness test (`npp/tests/golden_resize.rs`) verifies pixel-perfect
output for NearestNeighbor resize. On first GPU run it prints the captured output
and panics — copy the printed bytes into `EXPECTED` in that file to pin the
reference.

## Lint

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

## Documentation

```bash
cargo doc --no-deps -p npp-rs -p npp-sys
```

This builds crate-level docs with no broken intra-doc links. The `npp-rs` crate
has `#![deny(missing_docs)]` — missing doc comments are compile errors.

## What's Next

- See the [architecture guide](architecture.md) for how the crate is structured.
- See the [binding guide](npp-bindings.md) for how to add new NPP operations.
- See the [roadmap](roadmap.md) for post-M1 plans.
