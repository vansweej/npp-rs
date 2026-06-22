use std::collections::HashSet;

/// A classified NPP symbol that fits the simple type×channel grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedSymbol {
    /// The raw FFI symbol, e.g. "nppiResize_8u_C3R".
    pub raw: String,
    /// Operation family, e.g. "Resize".
    pub op: String,
    /// NPP element-type token, e.g. "8u", "32f".
    pub type_token: String,
    /// Channel count (1, 3, or 4).
    pub channels: u8,
    /// Full variant suffix (e.g., "C3R", "C4R", "C4C3R").
    pub variant: String,
}

/// Classify NPP symbols into the type×channel grid.
///
/// Parameterized by operation family prefix (e.g., "nppiResize_", "nppiSwapChannels_"),
/// accepted standard channel variants (C1R, C3R, C4R), and optional custom variants
/// like `C4C3R` (4→3 conversion).
///
/// Standard variants are matched for channels in `accepted_channels`, **unless** a
/// custom variant exists for that same channel count — in which case the custom
/// variant replaces the standard one. This allows families like SwapChannels to use
/// `C4C3R` instead of `C4R`.
///
/// When both `C1R` and `C1R_Ctx` (or `C3R` and `C3R_Ctx`, etc.) exist for the same
/// type and channel count, the `_Ctx` variant is preferred. Variants with extra
/// keywords (SqrPixel, Batch, Advanced, AC4, P{n}, C2, Sfs) are rejected. Rejected
/// symbols are simply omitted from the returned `Vec`.
pub fn classify(
    symbols: &[&str],
    prefix: &str,
    accepted_channels: &[u8],
    custom_variants: &[(u8, &str)],
) -> Vec<ClassifiedSymbol> {
    let family = prefix
        .strip_prefix("nppi")
        .and_then(|s| s.strip_suffix("_"))
        .unwrap_or("");

    // Build a set of channels that have a custom variant — standard variants for
    // those channels are suppressed.
    let custom_chs: HashSet<u8> = custom_variants.iter().map(|(ch, _)| *ch).collect();

    // First pass: collect all valid candidates grouped by (type_token, channels)
    let mut candidates: std::collections::HashMap<(String, u8), Vec<(String, String)>> =
        std::collections::HashMap::new();

    for raw in symbols {
        let s = match raw.strip_prefix(prefix) {
            Some(s) => s,
            None => continue,
        };

        // Reject any symbol with extra keywords after the prefix
        if !s.starts_with(|c: char| c.is_ascii_digit() || c == 'f') {
            continue; // e.g. "nppiResizeSqrPixel_8u_C1R"
        }

        let (type_token, rest) = match s.split_once('_') {
            Some((t, r)) => (t, r),
            None => continue,
        };

        // Validate type_token
        if !["8u", "8s", "16u", "16s", "32u", "32s", "32f", "64f", "16f"].contains(&type_token) {
            continue;
        }

        // Try to match variant and extract channels
        let (channels, variant_base) = match rest {
            "C1R" if accepted_channels.contains(&1) && !custom_chs.contains(&1) => (1, "C1R"),
            "C1R_Ctx" if accepted_channels.contains(&1) && !custom_chs.contains(&1) => {
                (1, "C1R_Ctx")
            }
            "C3R" if accepted_channels.contains(&3) && !custom_chs.contains(&3) => (3, "C3R"),
            "C3R_Ctx" if accepted_channels.contains(&3) && !custom_chs.contains(&3) => {
                (3, "C3R_Ctx")
            }
            "C4R" if accepted_channels.contains(&4) && !custom_chs.contains(&4) => (4, "C4R"),
            "C4R_Ctx" if accepted_channels.contains(&4) && !custom_chs.contains(&4) => {
                (4, "C4R_Ctx")
            }
            // Custom variants matched by exact suffix (with optional _Ctx)
            s if custom_variants.iter().any(|(_, v)| *v == s) => {
                let ch = custom_variants.iter().find(|(_, v)| *v == s).unwrap().0;
                (ch, s)
            }
            s if s.ends_with("_Ctx") => {
                // Check if the base (without _Ctx) is a custom variant
                let base = &s[..s.len() - 4]; // strip "_Ctx"
                if custom_variants.iter().any(|(_, v)| *v == base) {
                    let ch = custom_variants.iter().find(|(_, v)| *v == base).unwrap().0;
                    (ch, s)
                } else {
                    continue; // unknown variant with _Ctx suffix
                }
            }
            _ => continue, // rejects AC4R, P3R, C2R, and other unknown variants
        };

        candidates
            .entry((type_token.to_string(), channels))
            .or_default()
            .push((variant_base.to_string(), raw.to_string()));
    }

    // Second pass: for each (type_token, channels) group, prefer _Ctx if it exists
    let mut result = Vec::new();
    for ((type_token, channels), variants) in candidates {
        // Prefer _Ctx variant if present
        let (variant, raw) =
            if let Some((v, r)) = variants.iter().find(|(v, _)| v.ends_with("_Ctx")) {
                (v.clone(), r.clone())
            } else {
                let (v, r) = &variants[0];
                (v.clone(), r.clone())
            };

        result.push(ClassifiedSymbol {
            raw,
            op: family.to_string(),
            type_token,
            channels,
            variant,
        });
    }

    result
}

/// Classify NPP Resize symbols into the type×channel grid.
///
/// Convenience wrapper for the Resize family (C1/C3/C4 only).
pub fn classify_resize(symbols: &[&str]) -> Vec<ClassifiedSymbol> {
    classify(symbols, "nppiResize_", &[1, 3, 4], &[])
}

/// Classify NPP SwapChannels symbols into the type×channel grid.
///
/// Convenience wrapper for the SwapChannels family (C4C3R: 4→3 conversion).
pub fn classify_swap_channels(symbols: &[&str]) -> Vec<ClassifiedSymbol> {
    classify(symbols, "nppiSwapChannels_", &[4], &[(4, "C4C3R")])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert classify returns exactly one match with expected fields.
    fn assert_classify_one_resize(symbol: &str, expected_type: &str, expected_ch: u8) {
        let result = classify_resize(&[symbol]);
        assert_eq!(result.len(), 1, "expected one match for {symbol}");
        assert_eq!(result[0].raw, symbol);
        assert_eq!(result[0].op, "Resize");
        assert_eq!(result[0].type_token, expected_type);
        assert_eq!(result[0].channels, expected_ch);
    }

    // ── Resize tests (original 18) ──

    #[test]
    fn accept_8u_c3r() {
        assert_classify_one_resize("nppiResize_8u_C3R", "8u", 3);
    }

    #[test]
    fn accept_16u_c1r() {
        assert_classify_one_resize("nppiResize_16u_C1R", "16u", 1);
    }

    #[test]
    fn accept_32f_c4r() {
        assert_classify_one_resize("nppiResize_32f_C4R", "32f", 4);
    }

    #[test]
    fn accept_8s_c3r() {
        assert_classify_one_resize("nppiResize_8s_C3R", "8s", 3);
    }

    #[test]
    fn accept_64f_c1r() {
        assert_classify_one_resize("nppiResize_64f_C1R", "64f", 1);
    }

    #[test]
    fn accept_16f_c3r() {
        assert_classify_one_resize("nppiResize_16f_C3R", "16f", 3);
    }

    #[test]
    fn reject_sqr_pixel() {
        assert!(classify_resize(&["nppiResizeSqrPixel_8u_C1R"]).is_empty());
    }

    #[test]
    fn reject_batch() {
        assert!(classify_resize(&["nppiResizeBatch_8u_C1R"]).is_empty());
    }

    #[test]
    fn reject_ac4() {
        assert!(classify_resize(&["nppiResize_8u_AC4R"]).is_empty());
    }

    #[test]
    fn reject_planar() {
        assert!(classify_resize(&["nppiResize_8u_P3R"]).is_empty());
    }

    #[test]
    fn reject_c2() {
        assert!(classify_resize(&["nppiResize_8u_C2R"]).is_empty());
    }

    #[test]
    fn reject_c4c3r() {
        assert!(classify_resize(&["nppiResize_8u_C4C3R"]).is_empty());
    }

    #[test]
    fn reject_sfs_suffix() {
        assert!(classify_resize(&["nppiResize_8u_C3RSfs"]).is_empty());
    }

    #[test]
    fn reject_no_channel() {
        assert!(classify_resize(&["nppiResize_32f"]).is_empty());
    }

    #[test]
    fn reject_tx_suffix() {
        assert!(classify_resize(&["nppiResize_8u_C3Rtx"]).is_empty());
    }

    #[test]
    fn reject_invalid_type_token() {
        assert!(classify_resize(&["nppiResize_zzz_C3R"]).is_empty());
    }

    #[test]
    fn fixture_all_classified_are_valid() {
        let text = include_str!("../tests/fixtures/nppiResize_symbols.txt");
        let symbols: Vec<&str> = text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(!symbols.is_empty(), "fixture must not be empty");
        let result = classify_resize(&symbols);
        assert!(
            !result.is_empty(),
            "classify must produce at least one result"
        );
        for cs in &result {
            assert!(
                cs.channels == 1 || cs.channels == 3 || cs.channels == 4,
                "channels must be 1, 3, or 4, got {}",
                cs.channels
            );
            assert!(
                ["8u", "8s", "16u", "16s", "32u", "32s", "32f", "64f", "16f"]
                    .contains(&cs.type_token.as_str()),
                "unknown type_token {}",
                cs.type_token
            );
        }
    }

    // ── _Ctx variant tests ──

    #[test]
    fn accept_8u_c1r_ctx() {
        let result = classify_resize(&["nppiResize_8u_C1R_Ctx"]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].raw, "nppiResize_8u_C1R_Ctx");
        assert_eq!(result[0].variant, "C1R_Ctx");
        assert_eq!(result[0].channels, 1);
    }

    #[test]
    fn accept_32f_c4r_ctx() {
        let result = classify_resize(&["nppiResize_32f_C4R_Ctx"]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].variant, "C4R_Ctx");
        assert_eq!(result[0].channels, 4);
    }

    #[test]
    fn prefer_ctx_over_non_ctx() {
        // When both C1R and C1R_Ctx exist, prefer _Ctx
        let result = classify_resize(&["nppiResize_8u_C1R", "nppiResize_8u_C1R_Ctx"]);
        assert_eq!(
            result.len(),
            1,
            "should produce one result per (type, channel) pair"
        );
        assert_eq!(result[0].variant, "C1R_Ctx", "should prefer _Ctx variant");
        assert_eq!(result[0].raw, "nppiResize_8u_C1R_Ctx");
    }

    #[test]
    fn fixture_prefers_ctx_variants() {
        // Verify that the fixture generates _Ctx variants when available
        let text = include_str!("../tests/fixtures/nppiResize_symbols.txt");
        let symbols: Vec<&str> = text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        let result = classify_resize(&symbols);
        // All results should have _Ctx variant (since fixture has both and we prefer _Ctx)
        for cs in &result {
            assert!(
                cs.variant.ends_with("_Ctx"),
                "fixture should produce _Ctx variants; got {}",
                cs.variant
            );
        }
    }

    // ── SwapChannels tests ──

    #[test]
    fn swap_channels_accept_8u_c4c3r() {
        let result = classify_swap_channels(&["nppiSwapChannels_8u_C4C3R"]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "SwapChannels");
        assert_eq!(result[0].type_token, "8u");
        assert_eq!(result[0].channels, 4);
        assert_eq!(result[0].variant, "C4C3R");
    }

    #[test]
    fn swap_channels_accept_32f_c4c3r() {
        let result = classify_swap_channels(&["nppiSwapChannels_32f_C4C3R"]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].channels, 4);
        assert_eq!(result[0].variant, "C4C3R");
    }

    #[test]
    fn swap_channels_reject_c3r() {
        // SwapChannels only accepts C4C3R (4→3 conversion)
        assert!(classify_swap_channels(&["nppiSwapChannels_8u_C3R"]).is_empty());
    }

    #[test]
    fn swap_channels_reject_c4r() {
        assert!(classify_swap_channels(&["nppiSwapChannels_8u_C4R"]).is_empty());
    }

    #[test]
    fn swap_channels_reject_c1r() {
        assert!(classify_swap_channels(&["nppiSwapChannels_8u_C1R"]).is_empty());
    }

    #[test]
    fn swap_channels_reject_c2r() {
        assert!(classify_swap_channels(&["nppiSwapChannels_8u_C2R"]).is_empty());
    }

    #[test]
    fn swap_channels_fixture_all_c4c3r() {
        let text = include_str!("../tests/fixtures/nppiSwapChannels_symbols.txt");
        let symbols: Vec<&str> = text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(!symbols.is_empty(), "fixture must not be empty");
        let result = classify_swap_channels(&symbols);
        assert!(
            !result.is_empty(),
            "classify must produce at least one result"
        );
        for cs in &result {
            assert_eq!(cs.channels, 4, "all SwapChannels must map to channels=4");
            assert_eq!(
                cs.variant, "C4C3R",
                "all SwapChannels must have variant C4C3R"
            );
        }
    }

    // ── Parameterized API tests ──

    #[test]
    fn parameterized_classify_resize() {
        let result = classify(&["nppiResize_8u_C3R"], "nppiResize_", &[1, 3, 4], &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "Resize");
    }

    #[test]
    fn parameterized_classify_swap_channels() {
        let result = classify(
            &["nppiSwapChannels_8u_C4C3R"],
            "nppiSwapChannels_",
            &[4],
            &[(4, "C4C3R")],
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "SwapChannels");
        assert_eq!(result[0].variant, "C4C3R");
    }

    #[test]
    fn parameterized_reject_unsupported_channels() {
        // SwapChannels with C1R should be rejected for [4]-only
        let result = classify(
            &["nppiSwapChannels_8u_C1R"],
            "nppiSwapChannels_",
            &[4],
            &[(4, "C4C3R")],
        );
        assert!(result.is_empty());
    }
}
