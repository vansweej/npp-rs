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
    /// When true, the family carries two type tokens (src+dst); the generator
    /// uses `classify_convert` and emits `impl_*_for!($src_ty, $dst_ty, "$src_tok",
    /// "$dst_tok", { … })`.
    pub dual_type: bool,
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

Otherwise (Resize, SwapChannels, Convert) it emits bare symbols:
```rust
impl_resize_for!(u8, "8u", {
    1 => npp_sys::nppiResize_8u_C1R,
    3 => npp_sys::nppiResize_8u_C3R,
    4 => npp_sys::nppiResize_8u_C4R,
});
```

## Dual-type families

The Convert family carries **two** type tokens (src + dst) instead of one, because
conversion happens between different pixel types (e.g. `u8 → f32`). This is
supported by the `dual_type: true` field on `FamilyDescriptor`.

When `dual_type` is true, `generate_for_family()` calls `classify_convert()`
instead of `classify()`. `classify_convert()` uses a two-token split algorithm:

1. The segment between `nppiConvert_` and the first `_` is split at every
   position `k` such that both `segment[..k]` and `segment[k..]` are valid NPP
   type tokens (from `["8u","8s","16u","16s","32u","32s","32f","64f"]`).
2. **Exactly one** valid split → use that `(src, dst)` pair.
3. **Zero** valid splits → reject the symbol.
4. **Two or more** valid splits → `debug_assert!` panic (ambiguity bug, build-time only).

The generator emits a dual-type invocation:
```rust
impl_convert_for!(u8, f32, "8u", "32f", {
    1 => npp_sys::nppiConvert_8u32f_C1R_Ctx,
    3 => npp_sys::nppiConvert_8u32f_C3R_Ctx,
    4 => npp_sys::nppiConvert_8u32f_C4R_Ctx,
});
```

## Families implemented

| Family | Prefix | Channels | Custom variants | Buffer prefix | Shape | Dual-type |
|--------|--------|----------|----------------|---------------|-------|-----------|
| Resize | `nppiResize_` | C1, C3, C4 | — | — | `SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP` | No |
| SwapChannels | `nppiSwapChannels_` | C4 (src) | `C4C3R` | — | `SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER` | No |
| Mean | `nppiMean_` | C1, C3, C4 | — | `nppiMeanGetBufferHostSize_` | `SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR` | No |
| Convert | `nppiConvert_` | C1, C3, C4 | — | — | `SRC+STEP, DST+STEP, SIZE` | Yes |
| Normalize | `nppiConvert_` (reused) | C1, C3, C4 | — | — | `SRC+STEP, DST+STEP, SIZE` | Yes (dual-type, reused from Convert) |
| ConvertRounded | `nppiConvert_` | C1, C3, C4 | — | — | `SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode` | Yes |

## ConvertRounded: narrowing with explicit rounding

**ConvertRounded** is the rounding-mode counterpart of `ConvertTo` — it handles
**narrowing** conversions (e.g. `f32 → u8`) that require an explicit
`NppRoundMode` parameter. It follows the same dual-type codegen pattern as
Convert, with three differences:

1. **Separate fixture and classifier.** The round-mode `nppiConvert_*` symbols
   share `{src}{dst}_CxR` names with the no-rounding Convert symbols already in
   `nppiConvert_symbols.txt`. To avoid colliding with `classify_convert` (used by
   both `CONVERT_FAMILY` and Normalize against the same fixture), ConvertRounded
   uses its own `classify_convert_round` function and a separate fixture file
   (`nppiConvertRound_symbols.txt`). The shape-check
   (`convert_round_syn_shape_check`) is the load-bearing runtime validation that
   a fixture symbol truly has the `MISC:NppRoundMode` parameter shape.

2. **The `dual_type_round` selector.** `FamilyDescriptor` carries a
   `dual_type_round: bool` field (default `false`). When `true`,
   `generate_for_family()` dispatches to `classify_convert_round` instead of
   `classify_convert`. The selector is applied at both call sites in
   `gen_impls.rs` (the main generator and `validate_symbols_against_bindings`).

3. **The macro injects the round-mode argument.** The `impl_convert_rounded_for!`
   macro emits a method with signature
   `fn convert_rounded(&self, dst: &mut CudaImage<Dst>, mode: RoundMode) -> ...`
   and inserts `$crate::convert_round_ops::round_mode(mode)` between the
   `NppiSize` parameter and the stream-context parameter in the NPP FFI call.
   The generator emits identical invocation syntax to `impl_convert_for!`.

Scaled rounding-mode variants (shape `..., MISC:NppRoundMode, CONST_SCALAR`) are
deferred to **F5.4**.

## Normalize: convert-then-scale

Unlike the other four families, **Normalize does not use `FamilyDescriptor` or
`generate_for_family()`**. It has its own standalone generator
(`generate_normalize_impls`) in `gen_impls.rs` because of its unique structure:

- Normalize is a **composite** operation: `nppiConvert_*` (to f32) followed by
  `nppiMulC_32f_*_Ctx` (in-place scale).
- The generator filters the **existing** Convert fixture (`nppiConvert_symbols.txt`)
  to pairs where `dst_token == "32f"` and the source has a defined integer
  denominator. No separate fixture is needed.
- The `impl_normalize_for!` macro has a **trivial signature**:
  `($src_ty, $denominator, $src_token)` — no channel arms or symbol tuples.
  The convert step calls `self.convert(dst)?` via the `ConvertTo` trait; the
  MulC scale step is hardcoded inside the macro body using
  `nppiMulC_32f_C*R_Ctx`, dispatching on `dst.channels()` with explicit
  per-arm arrays.
- The scale denominator is the source type's maximum positive representable
  value (Option B, resolved): 255 for `u8`, 65535 for `u16`, 32767 for `i16`.

## How to add a new family

1. **Create a fixture** at `npp-codegen/tests/fixtures/nppi{Family}_symbols.txt`
   containing the NPP symbols you want to generate for (one per line, comments
   with `#`).

2. **Add a `FamilyDescriptor`** in `npp-codegen/src/gen_impls.rs`:
   - If your family uses non‑standard channel variants (like `C4C3R`), set
     `custom_variants`.
   - If your family is a two‑call scratch‑buffer op (like Mean), set
     `get_buffer_host_size_prefix`.
   - If your family carries two type tokens (src + dst, like Convert), set
     `dual_type: true`. You will also need to add a `classify_*` function in
     `classify.rs` and a dual‑type macro with signature
     `($src_ty:ty, $dst_ty:ty, $src_token:expr, $dst_token:expr, { … })`.

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
