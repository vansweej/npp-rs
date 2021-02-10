extern crate bindgen;

fn cuda_include_path() -> String {
    let cudadir = match option_env!("CUDA_INSTALL_DIR") {
        Some(cuda_dir) => format!("{}/include", cuda_dir),
        None => "/usr/local/cuda/include".to_string(),
    };

    cudadir
}

fn cuda_configuration() {
    let cudadir = match option_env!("CUDA_INSTALL_DIR") {
        Some(cuda_dir) => cuda_dir,
        None => "/usr/local/cuda",
    };

    println!("cargo:rustc-link-search={}/lib", cudadir);
}

fn main() {
    // Tell cargo to tell rustc to link the cuda libraries
    cuda_configuration();
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=nppc_static");
    println!("cargo:rustc-link-lib=static=nppial_static");
    println!("cargo:rustc-link-lib=static=nppicc_static");
    println!("cargo:rustc-link-lib=static=nppidei_static");
    println!("cargo:rustc-link-lib=static=nppif_static");
    println!("cargo:rustc-link-lib=static=nppig_static");
    println!("cargo:rustc-link-lib=static=nppim_static");
    println!("cargo:rustc-link-lib=static=nppist_static");
    println!("cargo:rustc-link-lib=static=nppisu_static");
    println!("cargo:rustc-link-lib=static=nppitc_static");

    println!("cargo:rustc-link-lib=culibos");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // CUDA include files
        .clang_args(&["-I", &cuda_include_path()[..]])
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // do not generate doc comments
        .generate_comments(false)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}
