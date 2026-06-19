use std::env;
use std::path::PathBuf;

/// Read CUDA installation path: first try `CUDA_PATH` (set by Nix shellHook),
/// fall back to `CUDA_INSTALL_DIR` (for non-Nix local builds), then
/// `/usr/local/cuda` as a last resort.
fn cuda_path() -> String {
    std::env::var("CUDA_PATH")
        .or_else(|_| option_env!("CUDA_INSTALL_DIR").map(|s| s.to_string()).ok_or(()))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string())
}

fn cuda_include_path() -> String {
    format!("{}/include", cuda_path())
}

fn cuda_configuration() {
    println!("cargo:rustc-link-search=native={}/lib", cuda_path());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path());
    // NPP libraries are in a separate store path from cuda_cudart
    if let Ok(npp_lib) = std::env::var("NPP_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", npp_lib);
    }
}

fn cuda_link_libs() {
    // Dynamic NPP libraries (.so)
    println!("cargo:rustc-link-lib=dylib=nppc");
    println!("cargo:rustc-link-lib=dylib=nppial");
    println!("cargo:rustc-link-lib=dylib=nppicc");
    println!("cargo:rustc-link-lib=dylib=nppidei");
    println!("cargo:rustc-link-lib=dylib=nppif");
    println!("cargo:rustc-link-lib=dylib=nppig");
    println!("cargo:rustc-link-lib=dylib=nppim");
    println!("cargo:rustc-link-lib=dylib=nppist");
    println!("cargo:rustc-link-lib=dylib=nppisu");
    println!("cargo:rustc-link-lib=dylib=nppitc");
    // CUDA runtime (dynamic)
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn main() {
    cuda_configuration();
    cuda_link_libs();

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NPP_LIB_PATH");
    println!("cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS");

    // Build clang args from the CUDA include path and BINDGEN_EXTRA_CLANG_ARGS
    let mut clang_args: Vec<String> = vec!["-I".to_string(), cuda_include_path()];
    if let Ok(extra) = std::env::var("BINDGEN_EXTRA_CLANG_ARGS") {
        // Split on whitespace, handling quoted -I arguments
        clang_args.extend(extra.split_whitespace().map(|s| s.to_string()));
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&clang_args)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate_comments(false)
        .allowlist_function("nppi.*")
        .allowlist_type("Nppi.*")
        .allowlist_var("Nppi.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
