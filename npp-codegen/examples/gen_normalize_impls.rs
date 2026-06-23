//! Generate `npp/src/normalize_generated.rs` from the Convert fixture.
//!
//! Reuses `nppiConvert_symbols.txt` (the same fixture as Convert) because
//! Normalize is a derived operation — it only wraps integer→f32 Convert pairs
//! with a scale step.
//!
//! Run: `cargo run --example gen_normalize_impls`

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir
        .join("tests")
        .join("fixtures")
        .join("nppiConvert_symbols.txt");
    let (symbols, _) = npp_codegen::gen_impls::read_fixture(&fixture);

    let generated = npp_codegen::gen_impls::generate_normalize_impls(&symbols);

    let out_path = manifest_dir
        .parent()
        .unwrap()
        .join("npp")
        .join("src")
        .join("normalize_generated.rs");

    std::fs::write(&out_path, generated.as_bytes())
        .expect("failed to write normalize_generated.rs");

    eprintln!("Wrote {}", out_path.display());
}
