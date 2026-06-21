# npp-codegen Architecture

> **npp-codegen** is a build‑time crate that generates Rust source for the
> `npp-rs` safe wrapper. It has **zero runtime footprint** — it only runs during
> development (via `cargo run --example gen_*_impls`). Its output is **committed**
> to the repo so that `npp-rs` has no build‑script dependency on it.

## Crate layout

| Module | Purpose |
|--------|---------|
| `classify.rs` | Parse an NPP symbol string into its type‑token, channel count, and variant suffix |
| `shape.rs` | Derive a parameter‑shape string from a function signature (for syn cross‑check) |
| `gen_impls.rs` | `FamilyDescriptor` struct + `generate_for_family()` — the core generator |
| `main.rs` | The `survey_shapes` binary (shape‑survey across all `nppi*` symbols) |

## The `FamilyDescriptor`

Every operation family (Resize, SwapChannels, Mean, …) is described by a static
`FamilyDescriptor`:

```rust
pub struct FamilyDescriptor {
    /// NPP prefix, e.g. "nppiResize_"
    pub npp_prefix: &'static str,
    /// Accepted channel counts, e.g. &[1, 3, 4]
    pub accepted_channels: &'static [u8],
    /// Non-standard variant suffixes beyond C1R/C3R/C4R, e.g. (4, "C4C3R")
    pub custom_variants: &'static [(u8, &'static str)],
    pub macro_name: &'static str,
    pub rust_macro_path: &'static str,
    /// Shape string for syn cross‑check, e.g. "SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP"
    pub expected_shape: &'static str,
    pub skip_16f: bool,
    pub use_statements: &'static [&'static str],
    /// For two‑call scratch‑buffer ops (e.g. Mean), the prefix for the
    /// buffer‑size query function. When `Some`, the generator emits
    /// `($mean_sym, $buffer_sym)` tuples instead of bare symbol names.
    pub get_buffer_host_size_prefix: Option<&'static str>,
}
```

## How classification works

`classify()` parses a raw symbol like `nppiSwapChannels_8u_C4C3R` into a
`ClassifiedSymbol { type_token: "8u", channels: 4, variant: "C4C3R" }`.

**Standard variants** (`C1R`, `C3R`, `C4R`) are always checked against
`accepted_channels`. **Custom variants** (e.g. `C4C3R`) are matched by exact
suffix and — crucially — **suppress the standard variant for the same channel
count**. This prevents e.g. `nppiResize_8u_C4C3R` from being accepted (Resize
only has C1R/C3R/C4R).

## How generation works

`generate_for_family()`:
1. **Classifies** all symbols from the fixture file.
2. **Groups** by type‑token, collecting `(channel, variant_suffix)` pairs.
3. **Emits** `impl_*_for!` invocations.

When `get_buffer_host_size_prefix` is set (Mean), the generator emits tuples:
```rust
impl_mean_for!(u8, "8u", {
    1 => (npp_sys::nppiMean_8u_C1R, npp_sys::nppiMeanGetBufferHostSize_8u_C1R),
    3 => (npp_sys::nppiMean_8u_C3R, npp_sys::nppiMeanGetBufferHostSize_8u_C3R),
    4 => (npp_sys::nppiMean_8u_C4R, npp_sys::nppiMeanGetBufferHostSize_8u_C4R),
});
```

Otherwise (Resize, SwapChannels) it emits bare symbols:
```rust
impl_resize_for!(u8, "8u", {
    1 => npp_sys::nppiResize_8u_C1R,
    3 => npp_sys::nppiResize_8u_C3R,
    4 => npp_sys::nppiResize_8u_C4R,
});
```

## Families implemented

| Family | Prefix | Channels | Custom variants | Buffer prefix | Shape |
|--------|--------|----------|----------------|---------------|-------|
| Resize | `nppiResize_` | C1, C3, C4 | — | — | `SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP` |
| SwapChannels | `nppiSwapChannels_` | C4 (src) | `C4C3R` | — | `SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER` |
| Mean | `nppiMean_` | C1, C3, C4 | — | `nppiMeanGetBufferHostSize_` | `SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR` |

## How to add a new family

1. **Create a fixture** at `npp-codegen/tests/fixtures/nppi{Family}_symbols.txt`
   containing the NPP symbols you want to generate for (one per line, comments
   with `#`).

2. **Add a `FamilyDescriptor`** in `npp-codegen/src/gen_impls.rs`:
   - If your family uses non‑standard channel variants (like `C4C3R`), set
     `custom_variants`.
   - If your family is a two‑call scratch‑buffer op (like Mean), set
     `get_buffer_host_size_prefix`.

3. **Create a macro** at `npp/src/{family}_macros.rs` with the `impl_*_for!`
   macro definition. Follow the pattern in `resize_macros.rs` (single‑call) or
   `mean_macros.rs` (two‑call).

4. **Create a generator example** at
   `npp-codegen/examples/gen_{family}_impls.rs`.

5. **Generate** the output by running
   `cargo run --example gen_{family}_impls` with the Nix devshell.

6. **Wire up modules** in `npp/src/lib.rs` and add the trait definition in
   `npp/src/imageops.rs`.

7. **Add a golden test** (GPU‑gated) at `npp/tests/golden_{family}.rs`.

8. **Commit** the generated artifact (`*_generated.rs`), the fixture, the macro,
   the generator, and the golden test.
