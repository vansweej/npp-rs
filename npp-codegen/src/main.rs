//! Binary: survey_shapes
//!
//! Reads NPP's bindgen output (`bindings.rs`), parses it with `syn`, and
//! produces a shape histogram showing the distribution of NPP functions
//! across normalized parameter-role patterns.

use npp_codegen::gen_impls;
use npp_codegen::shape::derive_shape;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let bindings_path = find_bindings_rs();
    let content = fs::read_to_string(&bindings_path).expect("Failed to read bindings.rs");

    let file = syn::parse_file(&content).expect("Failed to parse bindings.rs as Rust");

    let mut functions = Vec::new();
    for item in &file.items {
        if let syn::Item::ForeignMod(foreign_mod) = item {
            for item in &foreign_mod.items {
                if let syn::ForeignItem::Fn(func) = item {
                    let ident = func.sig.ident.to_string();
                    if ident.starts_with("nppi") {
                        functions.push((ident, func.sig.inputs.clone()));
                    }
                }
            }
        }
    }

    // Classify functions by shape
    let mut shape_histogram: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut families: HashSet<String> = HashSet::new();
    let mut ctx_count = 0;

    for (full_name, params) in &functions {
        // Strip _Ctx suffix
        let base_name = full_name
            .strip_suffix("_Ctx")
            .unwrap_or(full_name)
            .to_string();

        if full_name.ends_with("_Ctx") {
            ctx_count += 1;
        }

        // Extract family (nppi prefix up to first _\d)
        if let Some(family) = extract_family(&base_name) {
            families.insert(family);
        }

        // Derive shape from parameters (using library module)
        let shape = derive_shape(params);

        shape_histogram.entry(shape).or_default().push(base_name);
    }

    // Deduplicate base functions (collapse _Ctx twins)
    let mut unique_functions: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (shape, funcs) in shape_histogram {
        let mut unique = Vec::new();
        for func in funcs {
            if !unique.contains(&func) {
                unique.push(func);
            }
        }
        unique_functions.insert(shape, unique);
    }

    // Count distinct functions
    let total_functions: usize = unique_functions.values().map(|v| v.len()).sum();

    // Print reports
    println!("== TOTALS ==");
    println!(
        "distinct functions (base, _Ctx collapsed) : {}",
        total_functions
    );
    println!("  ...of which have a _Ctx twin            : {}", ctx_count);
    println!(
        "distinct families                         : {}",
        families.len()
    );
    println!(
        "distinct shapes                           : {}",
        unique_functions.len()
    );
    println!();

    // Coverage curve
    let mut shape_counts: Vec<(usize, String)> = unique_functions
        .iter()
        .map(|(shape, funcs)| (funcs.len(), shape.clone()))
        .collect();
    shape_counts.sort_by(|a, b| b.0.cmp(&a.0));

    let top_5: usize = shape_counts.iter().take(5).map(|(c, _)| c).sum();
    let top_10: usize = shape_counts.iter().take(10).map(|(c, _)| c).sum();
    let top_15: usize = shape_counts.iter().take(15).map(|(c, _)| c).sum();
    let top_20: usize = shape_counts.iter().take(20).map(|(c, _)| c).sum();
    let top_30: usize = shape_counts.iter().take(30).map(|(c, _)| c).sum();

    println!("== COVERAGE CURVE ==");
    println!(
        "  top  5: {} / {}  ({:.1}%)",
        top_5,
        total_functions,
        (top_5 as f64 / total_functions as f64) * 100.0
    );
    println!(
        "  top 10: {} / {}  ({:.1}%)",
        top_10,
        total_functions,
        (top_10 as f64 / total_functions as f64) * 100.0
    );
    println!(
        "  top 15: {} / {}  ({:.1}%)",
        top_15,
        total_functions,
        (top_15 as f64 / total_functions as f64) * 100.0
    );
    println!(
        "  top 20: {} / {}  ({:.1}%)",
        top_20,
        total_functions,
        (top_20 as f64 / total_functions as f64) * 100.0
    );
    println!(
        "  top 30: {} / {}  ({:.1}%)",
        top_30,
        total_functions,
        (top_30 as f64 / total_functions as f64) * 100.0
    );
    println!();

    // Singleton tail
    let n1 = unique_functions.values().filter(|v| v.len() == 1).count();
    let n2 = unique_functions.values().filter(|v| v.len() == 2).count();
    let n3 = unique_functions.values().filter(|v| v.len() == 3).count();

    println!("== SINGLETON TAIL ==");
    println!("  shapes used by exactly 1 function: {}", n1);
    println!("  shapes used by exactly 2 functions: {}", n2);
    println!("  shapes used by exactly 3 functions: {}", n3);
    println!();

    // Shape histogram
    println!("== SHAPE HISTOGRAM ==");
    for (count, shape) in &shape_counts {
        println!("  {} | {} | {}", count, shape, shape);
    }
    println!();

    // Family→Shape table
    println!("== FAMILY→SHAPE TABLE (families using each shape) ==");
    let mut family_shape_map: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (shape, funcs) in &unique_functions {
        for func in funcs {
            if let Some(family) = extract_family(func) {
                *family_shape_map
                    .entry(shape.clone())
                    .or_default()
                    .entry(family)
                    .or_insert(0) += 1;
            }
        }
    }

    for (shape, families_map) in &family_shape_map {
        println!("  {}:", shape);
        for (family, count) in families_map {
            println!("    - {} ({})", family, count);
        }
    }
    println!();

    // Resize sanity check
    println!("== RESIZE SANITY CHECK ==");
    let resize_shape = find_function_shape(&unique_functions, "nppiResize_8u_C1R");
    if let Some(shape) = resize_shape {
        println!("  nppiResize_8u_C1R -> {}", shape);
        let expected = "SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP";
        if shape == expected {
            println!("  OK[!]");
        } else {
            println!("  MISMATCH[!]");
            eprintln!("Expected: {}", expected);
            eprintln!("Got: {}", shape);
            std::process::exit(1);
        }
    } else {
        println!("  nppiResize_8u_C1R not found");
        std::process::exit(1);
    }
}

/// Find bindings.rs path using the `gen_impls` resolver.
fn find_bindings_rs() -> PathBuf {
    if let Ok(path) = env::var("BINDINGS_RS") {
        return PathBuf::from(path);
    }

    if let Ok(first_arg) = env::args().nth(1).ok_or(()) {
        return PathBuf::from(first_arg);
    }

    // Use the gen_impls resolver (fs::read_dir + substring match) instead of
    // shelling out to external `find`.
    if let Some(path) = gen_impls::find_bindings_rs() {
        return path;
    }

    eprintln!("Error: Could not find bindings.rs");
    eprintln!("Run `cargo build -p npp-sys` first, or pass `BINDINGS_RS=/path/to/bindings.rs`");
    std::process::exit(1);
}

/// Extract family from function name (e.g., "nppiResize" from "nppiResize_8u_C1R")
fn extract_family(name: &str) -> Option<String> {
    if !name.starts_with("nppi") {
        return None;
    }
    // Match nppi[A-Za-z]+?_\d
    let mut family = String::from("nppi");
    let rest = &name[4..];
    for ch in rest.chars() {
        if ch.is_ascii_digit() {
            break;
        }
        family.push(ch);
    }
    if family.len() > 4 {
        Some(family)
    } else {
        None
    }
}

/// Find the shape for a specific function
fn find_function_shape(histogram: &BTreeMap<String, Vec<String>>, target: &str) -> Option<String> {
    for (shape, funcs) in histogram {
        if funcs.iter().any(|f| f == target) {
            return Some(shape.clone());
        }
    }
    None
}
