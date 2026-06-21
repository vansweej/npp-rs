//! Generate `npp/src/mean_generated.rs` from the fixture.
//!
//! Run: `cargo run --example gen_mean_impls`

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir
        .join("tests")
        .join("fixtures")
        .join("nppiMean_symbols.txt");
    let (symbols, _) = npp_codegen::gen_impls::read_fixture(&fixture);

    let generated =
        npp_codegen::gen_impls::generate_for_family(&npp_codegen::gen_impls::MEAN_FAMILY, &symbols);

    let out_path = manifest_dir
        .parent()
        .unwrap()
        .join("npp")
        .join("src")
        .join("mean_generated.rs");

    std::fs::write(&out_path, generated.as_bytes()).expect("failed to write mean_generated.rs");

    eprintln!("Wrote {}", out_path.display());
}
