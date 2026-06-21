/// Assert device output matches a golden reference, or panic with printable
/// bytes if the golden is not yet pinned.
///
/// # Manual pin procedure
///
/// 1. Run `cargo test --features gpu --test <your_test>`.
/// 2. It will print the captured output and panic ("golden reference not
///    yet pinned").
/// 3. Copy the printed byte literal into `EXPECTED` in your test.
/// 4. Re-run to confirm.
///
/// # Safety
///
/// The CUDA device handle must outlive all buffers created from it (C7).
///
/// # Type contract
///
/// Every macro-generated op family should land with ≥1 bit-exact golden test,
/// preferring integer/nearest paths to avoid FP variance.
#[cfg(feature = "gpu")]
pub fn assert_golden<T: PartialEq + std::fmt::Debug>(actual: &[T], expected: &[T], label: &str) {
    if expected.is_empty() {
        eprintln!("=== Golden reference NOT pinned for {label} ===");
        eprintln!("Captured output ({} bytes): {:?}", actual.len(), actual);
        panic!("golden reference not yet pinned for {label}");
    }
    assert_eq!(actual, expected, "pixel mismatch in {label}");
}
