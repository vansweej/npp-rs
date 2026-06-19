# npp-rs
![build](https://github.com/vansweej/npp-rs/actions/workflows/build.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/npp-rs)](https://crates.io/crates/npp-rs)

Safe Rust bindings for NVIDIA NPP **image** operations (not signal ops).

Built on `cudarc` for CUDA device management and `npp-sys` (bindgen) for NPP FFI.
Features a `NppPixelType` alphabet (~9 types) with capability-trait dispatch:
unsupported `(type, op)` pairs are compile-time errors. Host round-trip via
`Vec<T>` — no `image` crate in the core.

## Dependencies

- Nix with flakes (enter the shell: `nix develop .`)
- An NVIDIA GPU with the proprietary driver installed
- CUDA toolkit + NPP libraries — provided by the Nix dev shell

## Quick start

```bash
# Enter the dev shell
nix develop .

# Build all crates
cargo build

# Run unit tests (no GPU required)
cargo test --no-default-features

# Run GPU-dependent tests (requires hardware)
cargo test --features gpu

# Lint
cargo clippy -- -D warnings
cargo fmt --check

# Documentation
cargo doc --no-deps -p npp-rs -p npp-sys
```

## Project structure

| Directory | Cargo package | Purpose                                    |
|-----------|---------------|--------------------------------------------|
| `npp-sys/` | `npp-sys`     | Bindgen FFI to NPP image domain (`nppi*`)  |
| `npp/`     | `npp-rs`      | Safe wrapper with `CudaImage`, traits, ops |

## License

MIT

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the post-M1 plan (macro codegen,
image-rs boundary, signal ops, IPP bindings, full golden test suite, etc.).
