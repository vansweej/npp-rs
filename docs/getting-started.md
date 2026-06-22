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
cargo test -p npp-codegen -p npp-rs
```

These tests cover pure-logic items: layout computations, error handling, symbol
classification, and codegen cross-checks. They do not initialize a CUDA device
or allocate GPU memory.

### GPU-dependent tests (requires hardware)

```bash
cargo test --features gpu
```

These tests require an NVIDIA GPU with the proprietary driver. CI has no GPU lane;
this is a documented manual gate for GPU hosts.

### Golden-image tests

Three correctness tests verify pixel-perfect output:

| Test | What it proves | Status |
|------|----------------|--------|
| `golden_resize.rs` | NearestNeighbor resize for `u8` | Pinned |
| `golden_swap_channels.rs` | BGRA→RGB reorder for `u8` | Pinned |
| `golden_mean.rs` | Per-channel mean for `u8` | Pinned |

To pin or re-pin a golden reference, run on a GPU host:

```bash
cargo test --features gpu --test golden_<name> -- --nocapture
```

The test will print the captured output and panic — copy the printed values
into the `EXPECTED` constant in that file to pin the reference.

## Lint

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

## Coverage

```bash
nix develop . --command cargo tarpaulin
```

This runs `cargo tarpaulin` on the **non-GPU surface** only (pure-logic layout,
error handling, codegen tests). GPU/CUDA/FFI code is excluded via
`tarpaulin.toml` globs and `#[cfg(not(tarpaulin_include))]` annotations.
The gate is set at 90 % — anything below that fails the run.

## Documentation

```bash
cargo doc --no-deps -p npp-rs -p npp-sys
```

This builds crate-level docs with no broken intra-doc links. The `npp-rs` crate
has `#![deny(missing_docs)]` — missing doc comments are compile errors.

## What's Next

- See the [architecture guide](architecture.md) for how the crate is structured.
- See the [codegen guide](codegen-architecture.md) for how macro-generated ops work.
- See the [binding guide](npp-bindings.md) for how to add new NPP operations manually.
- See the [roadmap](roadmap.md) for post-M1 plans.
