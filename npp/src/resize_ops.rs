use crate::imageops::ResizeInterpolation;
use npp_sys::{
    NppiInterpolationMode_NPPI_INTER_CUBIC, NppiInterpolationMode_NPPI_INTER_LANCZOS,
    NppiInterpolationMode_NPPI_INTER_LINEAR, NppiInterpolationMode_NPPI_INTER_NN,
    NppiInterpolationMode_NPPI_INTER_SUPER,
};

pub(crate) fn interpolation_mode(inter: ResizeInterpolation) -> i32 {
    match inter {
        ResizeInterpolation::NearestNeighbor => NppiInterpolationMode_NPPI_INTER_NN as i32,
        ResizeInterpolation::Linear => NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
        ResizeInterpolation::Cubic => NppiInterpolationMode_NPPI_INTER_CUBIC as i32,
        ResizeInterpolation::Super => NppiInterpolationMode_NPPI_INTER_SUPER as i32,
        ResizeInterpolation::Lanczos => NppiInterpolationMode_NPPI_INTER_LANCZOS as i32,
    }
}

/// Returns true iff the probe confirmed `(type_token, inter)` is supported.
pub(crate) fn mode_supported(type_token: &str, inter: ResizeInterpolation) -> bool {
    crate::resize_caps::RESIZE_CAPS
        .iter()
        .any(|(t, m)| *t == type_token && *m == inter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-existent type token should never appear in RESIZE_CAPS.
    /// This tests the guard logic path without needing a GPU.
    #[test]
    fn test_mode_unsupported_returns_false() {
        assert!(!mode_supported("zzz", ResizeInterpolation::Super));
    }

    /// Known-supported pair from the probed table should return true.
    #[test]
    fn test_mode_supported_returns_true() {
        assert!(mode_supported("8u", ResizeInterpolation::NearestNeighbor));
    }
}
