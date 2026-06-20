/// Generator: reads the nppiResize symbol fixture and emits `impl_resize_for!`
/// invocations that should be pasted into `npp/src/resize_generated.rs`.
///
/// Run: `cargo run -p npp-rs --example gen_resize_impls > /tmp/resize_generated.rs`
use npp_rs::suffix_classifier::classify;
use std::collections::BTreeMap;

fn main() {
    let text = include_str!("../tests/fixtures/nppiResize_symbols.txt");

    let symbols: Vec<&str> = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let classified = classify(&symbols);

    // Group by type_token, collecting unique channel counts.
    let mut groups: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for cs in &classified {
        // Skip 16f (half crate disabled)
        if cs.type_token == "16f" {
            continue;
        }
        groups.entry(&cs.type_token).or_default().push(cs.channels);
    }

    // Map NPP type token to Rust type.
    let rust_ty = |token: &str| -> Option<&str> {
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
    };

    // Emit header guard comment
    println!("// GENERATED — re-run `cargo run --example gen_resize_impls` on CUDA bump.");
    println!("// This file is **committed** (like `resize_caps.rs`), not gitignored.");
    println!();

    for (token, channels) in &groups {
        let rty = match rust_ty(token) {
            Some(t) => t,
            None => continue,
        };

        // Deduplicate and sort channel counts
        let mut chs: Vec<u8> = channels.clone();
        chs.sort();
        chs.dedup();

        let arms: Vec<String> = chs
            .iter()
            .map(|ch| format!("        {ch} => npp_sys::nppiResize_{token}_C{ch}R,"))
            .collect();

        println!("impl_resize_for!({rty}, \"{token}\", {{");
        for arm in &arms {
            println!("{arm}");
        }
        println!("}});");
        println!();
    }
}
