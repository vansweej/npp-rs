//! Generator: reads a family symbol fixture and emits `impl_*_for!` invocations.
//!
//! This is the built-time counterpart of the runtime macros. It takes a family
//! descriptor (NPP prefix, accepted channels, expected shape, macro name) and
//! a fixture path, then emits the `impl_*_for!` invocations that should be
//! pasted into `npp/src/*_generated.rs`.

use crate::classify::{
    classify, classify_convert, classify_convert_round, classify_convert_round_scaled,
    CONVERT_ROUND_SCALED_SHAPE,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Descriptor for a single NPP op family.
#[derive(Debug, Clone)]
pub struct FamilyDescriptor {
    /// NPP prefix, e.g. `"nppiResize_"` or `"nppiSwapChannels_"`.
    pub npp_prefix: &'static str,
    /// Accepted channel counts.
    pub accepted_channels: &'static [u8],
    /// Custom variant suffixes beyond C1R/C3R/C4R, e.g. `(4, "C4C3R")`.
    pub custom_variants: &'static [(u8, &'static str)],
    /// Name of the Rust macro, e.g. `"impl_resize_for"` or `"impl_swap_channels_for"`.
    pub macro_name: &'static str,
    /// Rust macro path for invocation (prepended with `impl_` or equivalent).
    pub rust_macro_path: &'static str,
    /// Expected parameter shape string (for syn cross-check).
    pub expected_shape: &'static str,
    /// Whether to skip `16f` (half crate disabled).
    pub skip_16f: bool,
    /// Additional use statements to include in generated output.
    pub use_statements: &'static [&'static str],
    /// For two-call ops (e.g. Mean), the NPP prefix for the buffer-size query
    /// function. When set, the generator emits `($mean_sym, $buffer_sym)` tuples
    /// instead of bare symbol names. `None` for standard single-call ops.
    pub get_buffer_host_size_prefix: Option<&'static str>,
    /// When true, the family carries two type tokens (src+dst); the generator
    /// uses `classify_convert` and emits `impl_*_for!($src_ty, $dst_ty, "$src_tok",
    /// "$dst_tok", { … })`.
    pub dual_type: bool,
    /// When true, the dual-type generator path uses `classify_convert_round`
    /// instead of `classify_convert`. Default `false`.
    pub dual_type_round: bool,
    /// When true, the dual-type generator path uses `classify_convert_round_scaled`
    /// instead of `classify_convert`/`classify_convert_round`. The generated
    /// macro invocations emit `impl_convert_rounded_scaled_for!` with an extra
    /// `scale_factor` parameter. Default `false`.
    pub dual_type_round_scaled: bool,
    /// Optional prefix for the engine-function-name argument (e.g. `"resize_into_"`).
    /// When `Some`, the generator emits `impl_*_for!(<engine_name>, <type>, ...)` where
    /// `<engine_name>` = `{prefix}{type_token}`. When `None`, no engine name is emitted
    /// (compatible with macros that do not accept the leading argument).
    pub engine_fn_prefix: Option<&'static str>,
    /// The Cargo example name (without `gen_` prefix) used in the generated-file
    /// header "// Regenerate with:" comment (e.g. `"convert_impls"`, `"resize_impls"`).
    pub example_name: &'static str,
}

/// Descriptor for the NPP Resize family.
pub static RESIZE_FAMILY: FamilyDescriptor = FamilyDescriptor {
    npp_prefix: "nppiResize_",
    accepted_channels: &[1, 3, 4],
    custom_variants: &[],
    macro_name: "impl_resize_for",
    rust_macro_path: "impl_resize_for",
    expected_shape: "SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP",
    skip_16f: true,
    use_statements: &[
        "use crate::error::{check_status, NppError};",
        "use crate::image::CudaImage;",
        "use crate::imageops::{Resize, ResizeInterpolation};",
        "use crate::impl_resize_for;",
        "use npp_sys::{NppiRect, NppiSize};",
    ],
    get_buffer_host_size_prefix: None,
    dual_type: false,
    dual_type_round: false,
    dual_type_round_scaled: false,
    engine_fn_prefix: Some("resize_into_"),
    example_name: "resize_impls",
};

/// Descriptor for the NPP SwapChannels family.
pub static SWAP_CHANNELS_FAMILY: FamilyDescriptor = FamilyDescriptor {
    npp_prefix: "nppiSwapChannels_",
    accepted_channels: &[4],
    custom_variants: &[(4, "C4C3R")],
    macro_name: "impl_swap_channels_for",
    rust_macro_path: "impl_swap_channels_for",
    expected_shape: "SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER",
    skip_16f: true,
    use_statements: &[
        "use crate::error::{check_status, NppError};",
        "use crate::image::CudaImage;",
        "use crate::imageops::SwapChannels;",
        "use crate::impl_swap_channels_for;",
        "use npp_sys::NppiSize;",
    ],
    get_buffer_host_size_prefix: None,
    dual_type: false,
    dual_type_round: false,
    dual_type_round_scaled: false,
    engine_fn_prefix: Some("swap_into_"),
    example_name: "swap_channels_impls",
};

/// Descriptor for the NPP Mean family (two-call dance with scratch buffer).
pub static MEAN_FAMILY: FamilyDescriptor = FamilyDescriptor {
    npp_prefix: "nppiMean_",
    accepted_channels: &[1, 3, 4],
    custom_variants: &[],
    macro_name: "impl_mean_for",
    rust_macro_path: "impl_mean_for",
    expected_shape: "SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR",
    skip_16f: true,
    use_statements: &[
        "use crate::error::{check_status, NppError};",
        "use crate::image::CudaImage;",
        "use crate::imageops::Mean;",
        "use crate::impl_mean_for;",
        "use npp_sys::NppiSize;",
    ],
    get_buffer_host_size_prefix: Some("nppiMeanGetBufferHostSize_"),
    dual_type: false,
    dual_type_round: false,
    dual_type_round_scaled: false,
    engine_fn_prefix: None,
    example_name: "mean_impls",
};

/// Descriptor for the NPP Convert family (dual-type, no-rounding shape only).
///
/// Covers the **no-rounding** Convert shape (`SRC+STEP, DST+STEP, SIZE`) **only**.
/// Rounding-mode Convert variants (`NppRoundMode`) are handled by
/// [`CONVERT_ROUND_FAMILY`].
pub static CONVERT_FAMILY: FamilyDescriptor = FamilyDescriptor {
    npp_prefix: "nppiConvert_",
    accepted_channels: &[1, 3, 4],
    custom_variants: &[],
    macro_name: "impl_convert_for",
    rust_macro_path: "impl_convert_for",
    expected_shape: "SRC+STEP, DST+STEP, SIZE",
    skip_16f: true,
    use_statements: &[
        "use crate::error::{check_status, NppError};",
        "use crate::image::CudaImage;",
        "use crate::imageops::ConvertTo;",
        "use crate::impl_convert_for;",
        "use npp_sys::NppiSize;",
    ],
    get_buffer_host_size_prefix: None,
    dual_type: true,
    dual_type_round: false,
    dual_type_round_scaled: false,
    engine_fn_prefix: None,
    example_name: "convert_impls",
};

/// Descriptor for the NPP ConvertRounded family (rounding-mode, dual-type).
///
/// Covers the **rounding-mode** Convert shape (`SRC+STEP, DST+STEP, SIZE,
/// MISC:NppRoundMode`) — narrowing conversions like `f32 → u8` that require
/// an explicit `NppRoundMode` parameter.
///
/// Uses a separate classifier (`classify_convert_round`) to avoid colliding
/// with `CONVERT_FAMILY`'s `classify_convert` (the symbol names are identical).
/// The fixture is `nppiConvertRound_symbols.txt`.
pub static CONVERT_ROUND_FAMILY: FamilyDescriptor = FamilyDescriptor {
    npp_prefix: "nppiConvert_",
    accepted_channels: &[1, 3, 4],
    custom_variants: &[],
    macro_name: "impl_convert_rounded_for",
    rust_macro_path: "impl_convert_rounded_for",
    expected_shape: "SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode",
    skip_16f: true,
    use_statements: &[
        "use crate::error::{check_status, NppError};",
        "use crate::image::CudaImage;",
        "use crate::imageops::{ConvertRounded, RoundMode};",
        "use crate::impl_convert_rounded_for;",
        "use npp_sys::NppiSize;",
    ],
    get_buffer_host_size_prefix: None,
    dual_type: true,
    dual_type_round: true,
    dual_type_round_scaled: false,
    engine_fn_prefix: None,
    example_name: "convert_round_impls",
};

/// Descriptor for the NPP ConvertRoundedScaled family (scaled rounding-mode,
/// dual-type, single-channel only).
///
/// Covers the **scaled rounding-mode** Convert shape (`SRC+STEP, DST+STEP, SIZE,
/// MISC:NppRoundMode, CONST_SCALAR`) — narrowing/widening conversions with both
/// a round mode and a scale factor. Only single-channel (`C1RSfs`) — NPP does
/// not expose `C3RSfs`/`C4RSfs`.
///
/// Uses a separate classifier (`classify_convert_round_scaled`) to match only
/// `C1RSfs`/`C1RSfs_Ctx` variants. The fixture is `nppiConvertRoundScaled_symbols.txt`.
pub static CONVERT_ROUND_SCALED_FAMILY: FamilyDescriptor = FamilyDescriptor {
    npp_prefix: "nppiConvert_",
    accepted_channels: &[1],
    custom_variants: &[],
    macro_name: "impl_convert_rounded_scaled_for",
    rust_macro_path: "impl_convert_rounded_scaled_for",
    expected_shape: CONVERT_ROUND_SCALED_SHAPE,
    skip_16f: true,
    use_statements: &[
        "use crate::error::{check_status, NppError};",
        "use crate::image::CudaImage;",
        "use crate::imageops::{ConvertRoundedScaled, RoundMode};",
        "use crate::impl_convert_rounded_scaled_for;",
        "use npp_sys::NppiSize;",
    ],
    get_buffer_host_size_prefix: None,
    dual_type: true,
    dual_type_round: false,
    dual_type_round_scaled: true,
    engine_fn_prefix: None,
    example_name: "convert_round_scaled_impls",
};

/// Map NPP type token to Rust type.
pub fn npp_type_to_rust(token: &str) -> Option<&'static str> {
    match token {
        "8u" => Some("u8"),
        "8s" => Some("i8"),
        "16u" => Some("u16"),
        "16s" => Some("i16"),
        "32u" => Some("u32"),
        "32s" => Some("i32"),
        "32f" => Some("f32"),
        "64f" => Some("f64"),
        _ => None,
    }
}

/// Read symbols from a fixture file, returning (lines, full_text).
pub fn read_fixture(path: &Path) -> (Vec<String>, String) {
    let text = fs::read_to_string(path).expect("failed to read fixture file");
    let symbols: Vec<String> = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|s| s.to_string())
        .collect();
    (symbols, text)
}

/// Generate `impl_*_for!` invocations for a family from raw symbol strings.
///
/// Returns the generated Rust source as a string.
pub fn generate_for_family(family: &FamilyDescriptor, symbols: &[String]) -> String {
    let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();

    // Extract the family name from the prefix (e.g., "Resize" from "nppiResize_"
    // or "Convert" from "nppiConvert_") — used for symbol name construction below.
    let family_name = family
        .npp_prefix
        .strip_prefix("nppi")
        .and_then(|s| s.strip_suffix("_"))
        .unwrap_or("");

    // Emit header guard comment using the family's example_name field
    let mut output = String::new();
    output.push_str("//! GENERATED — re-run `cargo run --example gen_");
    output.push_str(family.example_name);
    output.push_str("` on CUDA bump.\n");
    output.push_str("//! This file is **committed** (like `resize_caps.rs`), not gitignored.\n");
    output.push('\n');

    // Emit use statements
    for stmt in family.use_statements {
        output.push_str(stmt);
        output.push('\n');
    }
    output.push('\n');

    let mut macro_blocks: Vec<String> = Vec::new();

    if family.dual_type {
        // ── Dual-type branch (Convert family) ──
        let classified = if family.dual_type_round_scaled {
            classify_convert_round_scaled(&symbol_refs)
        } else if family.dual_type_round {
            classify_convert_round(&symbol_refs)
        } else {
            classify_convert(&symbol_refs)
        };

        // Group by (type_token, dst_type_token), collecting (channel, variant) pairs.
        let mut dual_groups: BTreeMap<(&str, &str), BTreeMap<u8, String>> = BTreeMap::new();
        for cs in &classified {
            let dst_tok = cs
                .dst_type_token
                .as_deref()
                .expect("dual_type family must have dst_type_token");

            // skip_16f: check both src and dst tokens
            if family.skip_16f && (cs.type_token == "16f" || dst_tok == "16f") {
                continue;
            }

            dual_groups
                .entry((&cs.type_token, dst_tok))
                .or_default()
                .entry(cs.channels)
                .or_insert_with(|| cs.variant.clone());
        }

        for ((src_token, dst_token), ch_variants) in &dual_groups {
            let src_rty = match npp_type_to_rust(src_token) {
                Some(t) => t,
                None => continue,
            };
            let dst_rty = match npp_type_to_rust(dst_token) {
                Some(t) => t,
                None => continue,
            };

            let mut block = String::new();
            block.push_str(&format!(
                "{}!({}, {}, \"{}\", \"{}\", {{\n",
                family.rust_macro_path, src_rty, dst_rty, src_token, dst_token
            ));
            for ch in ch_variants.keys() {
                let variant = &ch_variants[ch];
                // variant already includes _Ctx (from classify_convert's _Ctx preference)
                let sym = format!(
                    "npp_sys::nppiConvert_{}{}_{}",
                    src_token, dst_token, variant
                );
                block.push_str(&format!("        {} => {},\n", ch, sym));
            }
            block.push_str("});");
            macro_blocks.push(block);
        }
    } else {
        // ── Single-type branch (Resize, SwapChannels, Mean) ──
        let classified = classify(
            &symbol_refs,
            family.npp_prefix,
            family.accepted_channels,
            family.custom_variants,
        );

        // Group by type_token, collecting (channel, variant) pairs.
        let mut groups: BTreeMap<&str, BTreeMap<u8, String>> = BTreeMap::new();
        for cs in &classified {
            if family.skip_16f && cs.type_token == "16f" {
                continue;
            }
            groups
                .entry(&cs.type_token)
                .or_default()
                .entry(cs.channels)
                .or_insert_with(|| cs.variant.clone());
        }

        for (token, ch_variants) in &groups {
            let rty = match npp_type_to_rust(token) {
                Some(t) => t,
                None => continue,
            };

            let mut block = String::new();
            if let Some(prefix) = family.engine_fn_prefix {
                let engine_name = format!("{prefix}{token}");
                block.push_str(&format!(
                    "{}!({}, {}, \"{}\", {{\n",
                    family.rust_macro_path, engine_name, rty, token
                ));
            } else {
                block.push_str(&format!(
                    "{}!({}, \"{}\", {{\n",
                    family.rust_macro_path, rty, token
                ));
            }
            for ch in ch_variants.keys() {
                let variant = &ch_variants[ch];
                let sym = format!("npp_sys::nppi{}_{}_{}", family_name, token, variant);
                if let Some(buf_prefix) = family.get_buffer_host_size_prefix {
                    // Two-call dance: emit (mean_sym, buffer_sym) tuple
                    let buf_prefix_clean = buf_prefix
                        .strip_prefix("nppi")
                        .and_then(|s| s.strip_suffix("_"))
                        .unwrap_or("");
                    let buffer_sym =
                        format!("npp_sys::nppi{}_{}_{}", buf_prefix_clean, token, variant);
                    block.push_str(&format!("        {} => ({}, {}),\n", ch, sym, buffer_sym));
                } else {
                    block.push_str(&format!("        {} => {},\n", ch, sym));
                }
            }
            block.push_str("});");
            macro_blocks.push(block);
        }
    }

    // Join macro blocks with a blank line between them, no trailing blank line
    for (i, block) in macro_blocks.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(block);
        output.push('\n');
    }

    output
}

/// Validate that each classified symbol's parameter types match the family's expected shape.
///
/// Reads `bindings.rs` via `syn`, finds each classified function, derives its
/// shape from the actual parameter types, and compares against
/// `family.expected_shape`. Returns a list of mismatches.
pub fn validate_symbols_against_bindings(
    family: &FamilyDescriptor,
    symbols: &[String],
    bindings_rs_path: &Path,
) -> Vec<String> {
    let content = fs::read_to_string(bindings_rs_path).expect("failed to read bindings.rs");
    let file = syn::parse_file(&content).expect("failed to parse bindings.rs");

    // Build a map of function name -> params
    let mut func_map: std::collections::HashMap<
        String,
        syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
    > = std::collections::HashMap::new();
    for item in &file.items {
        if let syn::Item::ForeignMod(foreign_mod) = item {
            for item in &foreign_mod.items {
                if let syn::ForeignItem::Fn(func) = item {
                    let name = func.sig.ident.to_string();
                    func_map.insert(name, func.sig.inputs.clone());
                }
            }
        }
    }

    let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
    let classified: Vec<crate::classify::ClassifiedSymbol> = if family.dual_type_round_scaled {
        crate::classify::classify_convert_round_scaled(&symbol_refs)
    } else if family.dual_type_round {
        crate::classify::classify_convert_round(&symbol_refs)
    } else if family.dual_type {
        crate::classify::classify_convert(&symbol_refs)
    } else {
        classify(
            &symbol_refs,
            family.npp_prefix,
            family.accepted_channels,
            family.custom_variants,
        )
    };

    let mut mismatches = Vec::new();

    for cs in &classified {
        // Find the non-Ctx version first, then try _Ctx
        let func_params = func_map.get(&cs.raw).or_else(|| {
            let ctx_name = format!("{}_Ctx", cs.raw);
            func_map.get(&ctx_name)
        });

        if let Some(params) = func_params {
            let actual_shape = crate::shape::derive_shape(params);
            if actual_shape != family.expected_shape {
                mismatches.push(format!(
                    "{}: expected shape '{}', got '{}'",
                    cs.raw, family.expected_shape, actual_shape
                ));
            }
        } else {
            mismatches.push(format!("{}: function not found in bindings.rs", cs.raw));
        }
    }

    mismatches
}

/// Return the f32 normalization denominator for an NPP integer type token.
///
/// The denominator is the maximum positive representable value for the type,
/// such that calling `MulC` with `1/denominator` maps the type's maximum
/// positive value to exactly `1.0`:
///
/// | Token | Type   | BITS | Formula         | Value            |
/// |-------|--------|------|-----------------|------------------|
/// | 8u    | `u8`   | 8    | `2^8 - 1`       | 255              |
/// | 8s    | `i8`   | 8    | `2^(8-1) - 1`   | 127              |
/// | 16u   | `u16`  | 16   | `2^16 - 1`      | 65535            |
/// | 16s   | `i16`  | 16   | `2^(16-1) - 1`  | 32767            |
/// | 32u   | `u32`  | 32   | `2^32 - 1`      | 4294967295       |
/// | 32s   | `i32`  | 32   | `2^(32-1) - 1`  | 2147483647       |
/// | 32f   | `f32`  | —    | —               | `None` (excluded)|
/// | 64f   | `f64`  | —    | —               | `None` (excluded)|
///
/// Float sources are excluded because there is no canonical denominator —
/// they are already in floating-point range. These tokens are returned as
/// `f64` to avoid precision loss in the internal lookup; the generator
/// formats them as `{val}_f32` string literals for the emitted source.
///
/// ## Precision note (forward-looking)
///
/// The u32 and i32 denominators (`4294967295`, `2147483647`) are **not exactly
/// representable as f32**, and emitting them as `4294967295.0_f32` would trigger
/// `clippy::excessive_precision`. They are included in this helper for
/// completeness but are **not emitted today** — only u8/u16/i16 have Convert→f32
/// symbols. If a future CUDA bump adds u32→f32 or i32→f32 Convert symbols, the
/// generator must emit those denominators as a `const` cast (`u32::MAX as f32`)
/// rather than a decimal float literal.
pub fn normalize_scale_denominator(token: &str) -> Option<f64> {
    match token {
        "8u" => Some(255.0),
        "8s" => Some(127.0),
        "16u" => Some(65535.0),
        "16s" => Some(32767.0),
        "32u" => Some(4_294_967_295.0),
        "32s" => Some(2_147_483_647.0),
        "32f" | "64f" => None,
        _ => None,
    }
}

/// Generate `impl_normalize_for!` invocations for all integer→f32 pairs.
///
/// Reads the Convert fixture (same symbols as `CONVERT_FAMILY`), classifies
/// them, filters to pairs where `dst_token == "32f"` and the source has a
/// defined `normalize_scale_denominator`, and emits one invocation per
/// source type. The macro signature is trivially self-contained — no channel
/// arms or symbol tuples needed because the convert step delegates to the
/// `ConvertTo` trait and the MulC symbols are hardcoded in the macro body.
///
/// Unlike the other four families, this does NOT use `generate_for_family()`.
/// The Normalize generator has a completely different emit shape.
pub fn generate_normalize_impls(symbols: &[String]) -> String {
    let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
    let classified = classify_convert(&symbol_refs);

    // Build a BTreeMap<src_token, Option<f64>> for deterministic ordering.
    let mut normalize_map: BTreeMap<&str, f64> = BTreeMap::new();
    for cs in &classified {
        let dst_tok = match cs.dst_type_token.as_deref() {
            Some(t) => t,
            None => continue,
        };
        // Filter to dst == "32f" only
        if dst_tok != "32f" {
            continue;
        }
        // Skip 16f
        if cs.type_token == "16f" || dst_tok == "16f" {
            continue;
        }
        // Map Rust type
        if npp_type_to_rust(&cs.type_token).is_none() {
            continue;
        }
        // Check denominator
        let denom = match normalize_scale_denominator(&cs.type_token) {
            Some(d) => d,
            None => continue,
        };
        normalize_map.entry(&cs.type_token).or_insert(denom);
    }

    // Emit header
    let mut output = String::new();
    output.push_str(
        "//! GENERATED — re-run `cargo run --example gen_normalize_impls` on CUDA bump.\n",
    );
    output.push_str("//! This file is **committed** (like `resize_caps.rs`), not gitignored.\n");
    output.push('\n');

    // Emit use statements
    output.push_str("use crate::error::{check_status, NppError};\n");
    output.push_str("use crate::image::CudaImage;\n");
    output.push_str("use crate::imageops::{ConvertTo, Normalize};\n");
    output.push_str("use crate::impl_normalize_for;\n");
    output.push('\n');

    // Emit invocations in sort order
    for (token, denom) in &normalize_map {
        let rty = npp_type_to_rust(token).expect("already validated above");
        let denom_str = format!("{denom:.1}_f32");
        output.push_str(&format!(
            "impl_normalize_for!({rty}, {denom_str}, \"{token}\");\n"
        ));
    }

    output
}

/// Find the path to bindings.rs in the build directory.
pub fn find_bindings_rs() -> Option<PathBuf> {
    // Try relative to CARGO_MANIFEST_DIR
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crate_dir = manifest_dir.parent()?;
    let target_dir = crate_dir.join("target");

    // Walk debug/build directories looking for bindings.rs from npp-sys
    if let Ok(entries) = fs::read_dir(target_dir.join("debug").join("build")) {
        for entry in entries.flatten() {
            let path = entry.path().join("out").join("bindings.rs");
            if path.exists() && entry.file_name().to_string_lossy().contains("npp-sys") {
                return Some(path);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn normalize_denominators() {
        assert_eq!(normalize_scale_denominator("8u"), Some(255.0));
        assert_eq!(normalize_scale_denominator("16u"), Some(65535.0));
        assert_eq!(normalize_scale_denominator("16s"), Some(32767.0));
        assert_eq!(normalize_scale_denominator("8s"), Some(127.0));
        assert_eq!(normalize_scale_denominator("32f"), None);
        assert_eq!(normalize_scale_denominator("64f"), None);
        assert_eq!(normalize_scale_denominator("zzz"), None);
    }

    #[test]
    fn generate_normalize_from_fixture() {
        let fixture = fixture_path("nppiConvert_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        assert!(!symbols.is_empty(), "Convert fixture must not be empty");
        let generated = generate_normalize_impls(&symbols);
        assert!(!generated.is_empty(), "generated output must not be empty");

        // Must contain u8→f32 (8u has denominator 255)
        assert!(generated.contains("impl_normalize_for!(u8, 255.0_f32, \"8u\");"));
        // Must contain u16→f32
        assert!(generated.contains("impl_normalize_for!(u16, 65535.0_f32, \"16u\");"));
        // Must contain i16→f32
        assert!(generated.contains("impl_normalize_for!(i16, 32767.0_f32, \"16s\");"));
        // Must NOT contain 16f / f16
        assert!(!generated.contains("16f"));
        assert!(!generated.contains("f16"));
        // Must NOT contain float source types
        assert!(!generated.contains("impl_normalize_for!(f32"));
        assert!(!generated.contains("impl_normalize_for!(f64"));
        // Must include a ConvertTo import (macro body calls self.convert() via trait)
        assert!(generated.contains("use crate::imageops::{ConvertTo, Normalize};"));
        // Must NOT include DevicePtrMut (macro uses fully-qualified path)
        assert!(!generated.contains("use cudarc::driver::DevicePtrMut;"));
    }

    #[test]
    fn normalize_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("normalize_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed normalize_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiConvert_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_normalize_impls(&symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "normalize_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn resize_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("resize_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed resize_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiResize_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_for_family(&RESIZE_FAMILY, &symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "resize_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn convert_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("convert_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed convert_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiConvert_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_for_family(&CONVERT_FAMILY, &symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "convert_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn swap_channels_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("swap_channels_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed swap_channels_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiSwapChannels_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_for_family(&SWAP_CHANNELS_FAMILY, &symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "swap_channels_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn mean_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("mean_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed mean_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiMean_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_for_family(&MEAN_FAMILY, &symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "mean_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn convert_round_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("convert_round_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed convert_round_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiConvertRound_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_for_family(&CONVERT_ROUND_FAMILY, &symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "convert_round_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn swap_channels_corpus_is_generatable() {
        // Verify SwapChannels fixture can be read and generates valid output
        let fixture = fixture_path("nppiSwapChannels_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        assert!(
            !symbols.is_empty(),
            "SwapChannels fixture must not be empty"
        );

        let generated = generate_for_family(&SWAP_CHANNELS_FAMILY, &symbols);
        assert!(!generated.is_empty(), "generated output must not be empty");

        // Verify it contains the expected types with C4C3R_Ctx variant
        // (the classifier prefers _Ctx over non-_Ctx when both exist)
        assert!(generated.contains("impl_swap_channels_for!(swap_into_8u, u8, \"8u\", {"));
        assert!(generated.contains("4 => npp_sys::nppiSwapChannels_8u_C4C3R_Ctx,"));
        assert!(generated.contains("impl_swap_channels_for!(swap_into_32f, f32, \"32f\", {"));
        assert!(generated.contains("4 => npp_sys::nppiSwapChannels_32f_C4C3R_Ctx,"));
    }

    #[test]
    fn swap_channels_accepts_c4c3r_rejects_others() {
        // Only C4C3R variants should be classified; C3R, C4R are rejected.
        // The fixture now has both C4C3R and C4C3R_Ctx for each type; the
        // classifier deduplicates and prefers _Ctx, so we get 5 results
        // (one per type: i16, u16, f32, i32, u8).
        let fixture = fixture_path("nppiSwapChannels_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);

        // Count classified symbols
        let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
        let classified =
            crate::classify::classify(&symbol_refs, "nppiSwapChannels_", &[4], &[(4, "C4C3R")]);

        // Should have 5 results (one per unique type after _Ctx dedup)
        assert_eq!(
            classified.len(),
            5,
            "Expected 5 unique (type, channels) pairs after _Ctx dedup, got {}",
            classified.len()
        );

        // Verify no C3R or C4R made it through
        assert!(
            !classified.iter().any(|c| c.variant == "C3R"),
            "C3R variant must be rejected"
        );
        assert!(
            !classified.iter().any(|c| c.variant == "C4R"),
            "C4R variant must be rejected"
        );

        // Verify all are C4C3R or C4C3R_Ctx
        assert!(
            classified
                .iter()
                .all(|c| c.variant == "C4C3R" || c.variant == "C4C3R_Ctx"),
            "All classified must be C4C3R or C4C3R_Ctx"
        );
    }

    #[test]
    fn resize_syn_shape_check() {
        // Verify that Resize symbols have the expected shape when checked via syn
        let bindings = find_bindings_rs();
        if let Some(bindings_path) = bindings {
            let fixture = fixture_path("nppiResize_symbols.txt");
            let (symbols, _) = read_fixture(&fixture);

            let mismatches =
                validate_symbols_against_bindings(&RESIZE_FAMILY, &symbols, &bindings_path);
            if !mismatches.is_empty() {
                for m in &mismatches {
                    eprintln!("Mismatch: {}", m);
                }
            }
            assert!(
                mismatches.is_empty(),
                "All Resize symbols must match expected shape; {} mismatches found",
                mismatches.len()
            );
        } else {
            eprintln!("bindings.rs not found — skipping syn shape check");
        }
    }

    #[test]
    fn swap_channels_syn_shape_check() {
        // Verify that SwapChannels symbols have the expected shape
        let bindings = find_bindings_rs();
        if let Some(bindings_path) = bindings {
            let fixture = fixture_path("nppiSwapChannels_symbols.txt");
            let (symbols, _) = read_fixture(&fixture);

            let mismatches =
                validate_symbols_against_bindings(&SWAP_CHANNELS_FAMILY, &symbols, &bindings_path);
            if !mismatches.is_empty() {
                for m in &mismatches {
                    eprintln!("Mismatch: {}", m);
                }
            }
            assert!(
                mismatches.is_empty(),
                "All SwapChannels symbols must match expected shape; {} mismatches found",
                mismatches.len()
            );
        } else {
            eprintln!("bindings.rs not found — skipping syn shape check");
        }
    }

    #[test]
    fn generate_resize_from_fixture() {
        let fixture = fixture_path("nppiResize_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        assert!(!symbols.is_empty(), "Resize fixture must not be empty");
        let generated = generate_for_family(&RESIZE_FAMILY, &symbols);
        assert!(generated.contains("impl_resize_for!(resize_into_8u, u8, \"8u\", {"));
        assert!(generated.contains("impl_resize_for!(resize_into_32f, f32, \"32f\", {"));
    }

    #[test]
    fn generate_convert_from_fixture() {
        let fixture = fixture_path("nppiConvert_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        assert!(!symbols.is_empty(), "Convert fixture must not be empty");
        let generated = generate_for_family(&CONVERT_FAMILY, &symbols);
        assert!(!generated.is_empty(), "generated output must not be empty");

        // Verify dual-type invocation for u8 -> f32
        assert!(generated.contains("impl_convert_for!(u8, f32, \"8u\", \"32f\", {"));
        // Verify integer-widening invocation
        assert!(generated.contains("impl_convert_for!(u8, u16, \"8u\", \"16u\", {"));
        // Verify _Ctx symbol paths (B2 guard)
        assert!(generated.contains("npp_sys::nppiConvert_8u32f_C3R_Ctx"));
        // Verify no 16f / f16 tokens (skip_16f)
        assert!(!generated.contains("16f"));
        assert!(!generated.contains("f16"));
    }

    #[test]
    fn convert_syn_shape_check() {
        // Verify that Convert symbols have the expected shape (SRC+STEP, DST+STEP, SIZE)
        let bindings = find_bindings_rs();
        if let Some(bindings_path) = bindings {
            let fixture = fixture_path("nppiConvert_symbols.txt");
            let (symbols, _) = read_fixture(&fixture);

            let mismatches =
                validate_symbols_against_bindings(&CONVERT_FAMILY, &symbols, &bindings_path);
            if !mismatches.is_empty() {
                for m in &mismatches {
                    eprintln!("Mismatch: {}", m);
                }
            }
            assert!(
                mismatches.is_empty(),
                "All Convert symbols must match expected shape; {} mismatches found",
                mismatches.len()
            );
        } else {
            eprintln!("bindings.rs not found — skipping syn shape check");
        }
    }

    #[test]
    fn generate_convert_round_from_fixture() {
        let fixture = fixture_path("nppiConvertRound_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        assert!(
            !symbols.is_empty(),
            "ConvertRound fixture must not be empty"
        );
        let generated = generate_for_family(&CONVERT_ROUND_FAMILY, &symbols);
        assert!(!generated.is_empty(), "generated output must not be empty");

        // Verify dual-type invocation for f32 -> u8
        assert!(generated.contains("impl_convert_rounded_for!(f32, u8, \"32f\", \"8u\", {"));
        // Verify _Ctx symbol paths
        assert!(generated.contains("npp_sys::nppiConvert_32f8u_C3R_Ctx"));
        // Verify no 16f tokens (skip_16f) — check for the type token as a
        // quoted string literal rather than a bare substring, because valid
        // symbols like `nppiConvert_32f16u_*` contain "f16" as a substring
        // within the concatenated src+dst token.
        assert!(!generated.contains("\"16f\""));
    }

    #[test]
    fn convert_round_syn_shape_check() {
        // Verify that ConvertRound symbols have the expected shape
        // (SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode)
        let bindings = find_bindings_rs();
        if let Some(bindings_path) = bindings {
            let fixture = fixture_path("nppiConvertRound_symbols.txt");
            let (symbols, _) = read_fixture(&fixture);

            let mismatches =
                validate_symbols_against_bindings(&CONVERT_ROUND_FAMILY, &symbols, &bindings_path);
            if !mismatches.is_empty() {
                for m in &mismatches {
                    eprintln!("Mismatch: {}", m);
                }
            }
            assert!(
                mismatches.is_empty(),
                "All ConvertRound symbols must match expected shape; {} mismatches found",
                mismatches.len()
            );
        } else {
            eprintln!("bindings.rs not found — skipping syn shape check");
        }
    }

    // ── ConvertRoundedScaled (dual-type, round-mode, scaled) tests ──

    #[test]
    fn convert_round_scaled_generated_is_byte_identical() {
        // Read the committed generated file
        let committed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("npp")
            .join("src")
            .join("convert_round_scaled_generated.rs");
        let committed = fs::read_to_string(&committed_path)
            .expect("failed to read committed convert_round_scaled_generated.rs");

        // Generate from the fixture
        let fixture = fixture_path("nppiConvertRoundScaled_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        let generated = generate_for_family(&CONVERT_ROUND_SCALED_FAMILY, &symbols);

        // Show diff if not identical
        if committed != generated {
            eprintln!(
                "Committed length: {}, Generated length: {}",
                committed.len(),
                generated.len()
            );
            for (i, (cl, gl)) in committed.lines().zip(generated.lines()).enumerate() {
                if cl != gl {
                    eprintln!("First difference at line {}:", i + 1);
                    eprintln!("  Committed: {:?}", cl);
                    eprintln!("  Generated: {:?}", gl);
                    break;
                }
            }
        }

        assert_eq!(
            committed, generated,
            "convert_round_scaled_generated.rs must be byte-identical"
        );
    }

    #[test]
    fn convert_round_scaled_corpus_is_generatable() {
        // Verify that at least one C1RSfs symbol reaches the classifier from bindings.rs
        let bindings = find_bindings_rs();
        if let Some(bindings_path) = bindings {
            let content = fs::read_to_string(&bindings_path).expect("failed to read bindings.rs");

            // Find all C1RSfs symbols in bindings.rs
            let symbols: Vec<String> = content
                .lines()
                .filter(|l| l.contains("C1RSfs") && l.starts_with("nppiConvert"))
                .map(|l| {
                    // Extract the function name (before the first '(')
                    l.split_once('(')
                        .map(|(name, _)| name.trim().to_string())
                        .unwrap_or_default()
                })
                .filter(|s| !s.is_empty())
                .collect();

            if symbols.is_empty() {
                eprintln!("No C1RSfs symbols found in bindings.rs — skipping corpus test");
                return;
            }

            let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
            let classified = classify_convert_round_scaled(&symbol_refs);
            assert!(
                !classified.is_empty(),
                "classify_convert_round_scaled must classify at least one symbol from live bindings"
            );
        } else {
            eprintln!("bindings.rs not found — skipping corpus test");
        }
    }

    #[test]
    fn generate_convert_round_scaled_from_fixture() {
        let fixture = fixture_path("nppiConvertRoundScaled_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);
        assert!(
            !symbols.is_empty(),
            "ConvertRoundScaled fixture must not be empty"
        );
        let generated = generate_for_family(&CONVERT_ROUND_SCALED_FAMILY, &symbols);
        assert!(!generated.is_empty(), "generated output must not be empty");

        // Verify dual-type invocation for u8 -> i8
        assert!(generated.contains("impl_convert_rounded_scaled_for!(u8, i8, \"8u\", \"8s\", {"));
        // Verify _Ctx symbol paths (C1RSfs only — single channel)
        assert!(generated.contains("npp_sys::nppiConvert_8u8s_C1RSfs_Ctx"));
        // Verify no 16f tokens (skip_16f)
        assert!(!generated.contains("\"16f\""));
        // Verify single-channel only — no C3 or C4 arms
        assert!(!generated.contains("C3R"));
        assert!(!generated.contains("C4R"));
    }
}
