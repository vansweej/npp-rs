//! Helper to translate the safe `RoundMode` enum to an NPP `NppRoundMode` constant.
//!
//! `NppRoundMode` is a `c_uint` (= `u32` in Rust) in the bindgen output. This
//! function returns the raw constant value without casting, matching the
//! parameter type expected by the round-mode `nppiConvert_*` FFI functions.

use crate::imageops::RoundMode;

pub(crate) fn round_mode(mode: RoundMode) -> npp_sys::NppRoundMode {
    match mode {
        RoundMode::Nearest => npp_sys::NppRoundMode_NPP_RND_NEAR,
        RoundMode::Financial => npp_sys::NppRoundMode_NPP_RND_FINANCIAL,
        RoundMode::Zero => npp_sys::NppRoundMode_NPP_RND_ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_mode_nearest() {
        assert_eq!(
            round_mode(RoundMode::Nearest),
            npp_sys::NppRoundMode_NPP_RND_NEAR,
        );
    }

    #[test]
    fn test_round_mode_financial() {
        assert_eq!(
            round_mode(RoundMode::Financial),
            npp_sys::NppRoundMode_NPP_RND_FINANCIAL,
        );
    }

    #[test]
    fn test_round_mode_zero() {
        assert_eq!(
            round_mode(RoundMode::Zero),
            npp_sys::NppRoundMode_NPP_RND_ZERO,
        );
    }
}
