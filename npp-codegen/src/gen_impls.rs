//! Generator: reads a family symbol fixture and emits `impl_*_for!` invocations.
//!
//! This is the built-time counterpart of the runtime macros. It takes a family
//! descriptor (NPP prefix, accepted channels, expected shape, macro name) and
//! a fixture path, then emits the `impl_*_for!` invocations that should be
//! pasted into `npp/src/*_generated.rs`.

use crate::classify::{classify, classify_convert};
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
};

/// Descriptor for the NPP Convert family (dual-type, no-rounding shape only).
///
/// Covers the **no-rounding** Convert shape (`SRC+STEP, DST+STEP, SIZE`) **only**.
/// Rounding-mode Convert variants (`NppRoundMode`) are **deferred to F5.2**.
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
    // or "Convert" from "nppiConvert_")
    let family_name = family
        .npp_prefix
        .strip_prefix("nppi")
        .and_then(|s| s.strip_suffix("_"))
        .unwrap_or("");

    // Emit header guard comment
    let mut output = String::new();
    output.push_str("//! GENERATED — re-run `cargo run --example gen_");
    output.push_str(&family_name.to_lowercase());
    output.push_str("_impls` on CUDA bump.\n");
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
        let classified = classify_convert(&symbol_refs);

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
            block.push_str(&format!(
                "{}!({}, \"{}\", {{\n",
                family.rust_macro_path, rty, token
            ));
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
    let classified: Vec<crate::classify::ClassifiedSymbol> = if family.dual_type {
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

        // Verify it contains the expected types with C4C3R variant
        assert!(generated.contains("impl_swap_channels_for!(u8, \"8u\", {"));
        assert!(generated.contains("4 => npp_sys::nppiSwapChannels_8u_C4C3R,"));
        assert!(generated.contains("impl_swap_channels_for!(f32, \"32f\", {"));
        assert!(generated.contains("4 => npp_sys::nppiSwapChannels_32f_C4C3R,"));
    }

    #[test]
    fn swap_channels_accepts_c4c3r_rejects_others() {
        // Only C4C3R should be classified; C3R, C4R, _Ctx variants are rejected
        let fixture = fixture_path("nppiSwapChannels_symbols.txt");
        let (symbols, _) = read_fixture(&fixture);

        // Count classified symbols
        let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
        let classified =
            crate::classify::classify(&symbol_refs, "nppiSwapChannels_", &[4], &[(4, "C4C3R")]);

        // Should have exactly all the listed symbols (they're all C4C3R)
        assert_eq!(
            classified.len(),
            symbols.len(),
            "All {} symbols in fixture should be classified as C4C3R",
            symbols.len()
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

        // Verify all are C4C3R
        assert!(
            classified.iter().all(|c| c.variant == "C4C3R"),
            "All classified must be C4C3R"
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
        assert!(generated.contains("impl_resize_for!(u8, \"8u\", {"));
        assert!(generated.contains("impl_resize_for!(f32, \"32f\", {"));
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
}
