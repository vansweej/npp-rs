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
}

/// Classify NPP Resize symbols into the type×channel grid.
///
/// Only `nppiResize_<type>_C<n>R` (C1/C3/C4 only) is accepted.
/// SqrPixel, Batch, Advanced, AC4, P{n}, C2, Sfs, Ctx variants are rejected.
/// Rejected symbols are simply omitted from the returned `Vec`.
pub fn classify(symbols: &[&str]) -> Vec<ClassifiedSymbol> {
    symbols
        .iter()
        .filter_map(|raw| {
            let s = raw.strip_prefix("nppiResize_")?;
            // Reject any symbol with extra keywords after "Resize_"
            if !s.starts_with(|c: char| c.is_ascii_digit() || c == 'f') {
                return None; // e.g. "nppiResizeSqrPixel_8u_C1R"
            }
            let (type_token, rest) = s.split_once('_')?;
            // Validate type_token
            if !["8u", "8s", "16u", "16s", "32u", "32s", "32f", "64f", "16f"].contains(&type_token)
            {
                return None;
            }
            // Channel part must be C1, C3, or C4, ending with R
            let channels = match rest {
                "C1R" => 1,
                "C3R" => 3,
                "C4R" => 4,
                _ => return None, // rejects AC4R, P3R, C2R, C4C3R, C3RSfs, Ctx, etc.
            };
            Some(ClassifiedSymbol {
                raw: raw.to_string(),
                op: "Resize".to_string(),
                type_token: type_token.to_string(),
                channels,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert classify returns exactly one match with expected fields.
    fn assert_classify_one(symbol: &str, expected_type: &str, expected_ch: u8) {
        let result = classify(&[symbol]);
        assert_eq!(result.len(), 1, "expected one match for {symbol}");
        assert_eq!(result[0].raw, symbol);
        assert_eq!(result[0].op, "Resize");
        assert_eq!(result[0].type_token, expected_type);
        assert_eq!(result[0].channels, expected_ch);
    }

    #[test]
    fn accept_8u_c3r() {
        assert_classify_one("nppiResize_8u_C3R", "8u", 3);
    }

    #[test]
    fn accept_16u_c1r() {
        assert_classify_one("nppiResize_16u_C1R", "16u", 1);
    }

    #[test]
    fn accept_32f_c4r() {
        assert_classify_one("nppiResize_32f_C4R", "32f", 4);
    }

    #[test]
    fn accept_8s_c3r() {
        assert_classify_one("nppiResize_8s_C3R", "8s", 3);
    }

    #[test]
    fn accept_64f_c1r() {
        assert_classify_one("nppiResize_64f_C1R", "64f", 1);
    }

    #[test]
    fn accept_16f_c3r() {
        assert_classify_one("nppiResize_16f_C3R", "16f", 3);
    }

    #[test]
    fn reject_sqr_pixel() {
        assert!(classify(&["nppiResizeSqrPixel_8u_C1R"]).is_empty());
    }

    #[test]
    fn reject_batch() {
        assert!(classify(&["nppiResizeBatch_8u_C1R"]).is_empty());
    }

    #[test]
    fn reject_ac4() {
        assert!(classify(&["nppiResize_8u_AC4R"]).is_empty());
    }

    #[test]
    fn reject_planar() {
        assert!(classify(&["nppiResize_8u_P3R"]).is_empty());
    }

    #[test]
    fn reject_c2() {
        assert!(classify(&["nppiResize_8u_C2R"]).is_empty());
    }

    #[test]
    fn reject_c4c3r() {
        assert!(classify(&["nppiResize_8u_C4C3R"]).is_empty());
    }

    #[test]
    fn reject_sfs_suffix() {
        assert!(classify(&["nppiResize_8u_C3RSfs"]).is_empty());
    }

    #[test]
    fn reject_no_channel() {
        assert!(classify(&["nppiResize_32f"]).is_empty());
    }

    #[test]
    fn reject_tx_suffix() {
        assert!(classify(&["nppiResize_8u_C3Rtx"]).is_empty());
    }

    #[test]
    fn reject_invalid_type_token() {
        assert!(classify(&["nppiResize_zzz_C3R"]).is_empty());
    }

    #[test]
    fn fixture_all_classified_are_valid() {
        let text = include_str!("../tests/fixtures/nppiResize_symbols.txt");
        let symbols: Vec<&str> = text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(!symbols.is_empty(), "fixture must not be empty");
        let result = classify(&symbols);
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
}
