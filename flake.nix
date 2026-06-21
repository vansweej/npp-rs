{
  # ============================================================================
  # Reproducible Rust + CUDA + NPP development environment
  #
  # All build tools (Rust toolchain, CUDA toolkit components) are pinned via
  # flake.lock to exact nixpkgs and rust-overlay commits. This means the
  # environment can be reproduced exactly, even years from now, with:
  #
  #   nix develop
  #
  # The one unavoidable system dependency is the NVIDIA kernel driver interface
  # (libcuda.so.1 and friends). These libraries MUST match the kernel module
  # version installed on the host and cannot be packaged in Nix. Everything
  # else -- headers, NPP libraries, the Rust toolchain -- is fully Nix-managed.
  #
  # To update all pinned inputs:
  #
  #   nix flake update
  # ============================================================================

  description = "Reproducible Rust + CUDA development environment for npp-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true; # required for CUDA packages
        };

        rustVersion = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        cudaPackages = pkgs.cudaPackages;

        # ── System NVIDIA driver libs ─────────────────────────────────────────
        systemNvidiaLibDir = "/usr/lib/x86_64-linux-gnu";
        systemNvidiaLibs = [
          "libcuda.so.1"                   # CUDA driver API
          "libnvidia-ptxjitcompiler.so.1"  # PTX JIT compiler
          "libnvidia-nvvm.so.4"            # NVVM IR compiler
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          name = "npp-rs";

          buildInputs = with pkgs; [
            # Rust toolchain -- version pinned via rust-toolchain.toml
            rustVersion

            # LSP
            rust-analyzer

            # CUDA toolkit components (all pinned via flake.lock)
            cudaPackages.cuda_cudart  # CUDA runtime headers + stub libs
            cudaPackages.libnpp       # NPP shared libs + headers
            cudaPackages.cuda_nvrtc   # NVRTC runtime lib (needed by cudarc at link time)

            # libclang for bindgen
            llvmPackages.libclang

            # C++ standard library needed by CUDA headers
            stdenv.cc.cc.lib

            pkg-config

            # Coverage tool
            cargo-tarpaulin
          ];

          shellHook = ''
            export CUDA_PATH="${cudaPackages.cuda_cudart}"
            export NPP_LIB_PATH="${cudaPackages.libnpp.lib}/lib"

            # Expose libclang for bindgen
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"

            # Extra clang args for bindgen to find NPP/CUDA headers
            # libnpp has a multi-output derivation; headers are in the "include" output,
            # libraries in the "lib" output.
            # cuda_nvcc headers (crt/*.h) are needed because cuda_runtime.h includes
            # <crt/host_config.h> — this is a header dependency, not a compiler one.
            export BINDGEN_EXTRA_CLANG_ARGS="-I${cudaPackages.cuda_cudart}/include -I${cudaPackages.cuda_nvcc}/include -I${cudaPackages.libnpp.include}/include"

            # Expose Nix CUDA and C++ runtime libs for dynamic linking at runtime.
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${cudaPackages.cuda_cudart}/lib:${cudaPackages.libnpp.lib}/lib:$LD_LIBRARY_PATH"

            # NIX_LDFLAGS is set by mkShell and includes a -rpath $out/lib entry
            # where $out resolves to outputs/out inside the project directory.
            # That directory never exists and would be a dangling rpath entry in
            # every compiled binary. We remove only the two tokens "-rpath <out>"
            # while keeping all -L flags (which crtbeginS.o and friends need).
            export NIX_LDFLAGS="$(echo "$NIX_LDFLAGS" | sed 's|-rpath [^ ]*outputs/out[^ ]*||g')"

            # Create .nvidia-libs/ containing symlinks to the host NVIDIA driver
            # libs. This directory is added as an rpath in compiled binaries so
            # they find the real driver at runtime.
            mkdir -p .nvidia-libs
            for lib in ${pkgs.lib.concatStringsSep " " systemNvidiaLibs}; do
              src="${systemNvidiaLibDir}/$lib"
              if [ -f "$src" ]; then
                ln -sf "$src" ".nvidia-libs/$lib"
              fi
            done

            # Use RUSTFLAGS and LIBRARY_PATH for linker and library search paths.
            # RUSTFLAGS: single-token flags (safe for tarpaulin's direct rustc probe).
            # LIBRARY_PATH: search paths as env var (not subject to tarpaulin's
            # argument-parsing bug with space-separated -L flags).
            export RUSTFLAGS="-C link-arg=-Wl,-rpath,$PWD/.nvidia-libs -C link-arg=-Wl,-rpath,${pkgs.glibc}/lib $RUSTFLAGS"
            export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${cudaPackages.cuda_cudart}/lib/stubs:${cudaPackages.cuda_nvrtc}/lib:$LIBRARY_PATH"
          '';
        };
      }
    );
}
