use thiserror::Error;

/// Raw NPP status code. Positive values are warnings, zero is success,
/// negative values are errors.
pub type NppStatus = i32;

/// Errors that can occur during NPP operations.
#[derive(Error, Debug)]
pub enum NppError {
    /// NPP library returned a negative (error) status code.
    #[error("NPP returned error status {0}")]
    Npp(NppStatus),

    /// CUDA driver-level error (allocation, copy, context).
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    /// Invalid argument passed to a function (precondition failure).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Check an NPP status code and return `Ok(())` on success (zero or positive
/// warning code) or `Err(NppError::Npp(status))` on error (negative code).
///
/// # Notes
///
/// Positive `NppStatus` values are *warnings* and do not indicate failure.
/// This is a deliberate fix for the original crate's `status == 0` check which
/// incorrectly treated positive warning codes as hard errors (C1/NEW-01).
pub fn check_status(status: NppStatus) -> Result<(), NppError> {
    if status >= 0 {
        Ok(())
    } else {
        Err(NppError::Npp(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_status_ok_zero() {
        assert!(check_status(0).is_ok());
    }

    #[test]
    fn test_check_status_ok_positive_warning() {
        assert!(check_status(1).is_ok());
    }

    #[test]
    fn test_check_status_error_negative() {
        let err = check_status(-22).unwrap_err();
        assert_eq!(format!("{}", err), "NPP returned error status -22");
    }
}
