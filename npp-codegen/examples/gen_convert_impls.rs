//! Generate `npp/src/convert_generated.rs` from the fixture.
//!
//! Run: `cargo run --example gen_convert_impls`

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir
        .join("tests")
        .join("fixtures")
        .join("nppiConvert_symbols.txt");
    let (symbols, _) = npp_codegen::gen_impls::read_fixture(&fixture);

    let generated = npp_codegen::gen_impls::generate_for_family(
        &npp_codegen::gen_impls::CONVERT_FAMILY,
        &symbols,
    );

    let out_path = manifest_dir
        .parent()
        .unwrap()
        .join("npp")
        .join("src")
        .join("convert_generated.rs");

    std::fs::write(&out_path, generated.as_bytes()).expect("failed to write convert_generated.rs");

    eprintln!("Wrote {}", out_path.display());
}
