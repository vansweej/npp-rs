//! Shape derivation from NPP function signatures.
//!
//! Reads a function's parameter list and produces a normalized shape string
//! (e.g. `"SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP"`) by
//! classifying each parameter and merging adjacent pairs.

use syn;

/// Derive shape from function parameters
pub fn derive_shape(params: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>) -> String {
    let mut classified = Vec::new();

    for param in params {
        if let syn::FnArg::Typed(pat_type) = param {
            let param_name = extract_param_name(&pat_type.pat);
            let role = classify_param(&pat_type.ty, &pat_type.pat);
            if !role.is_empty() && role != "SKIP" {
                classified.push((param_name, role));
            }
        }
    }

    // Merge adjacent parameters based on naming patterns
    let mut merged = Vec::new();
    let mut i = 0;
    while i < classified.len() {
        let (name, role) = &classified[i];

        // Check for SRC+STEP pattern: pSrc/pSource followed by nSrcStep/nSourceStep
        if (name.starts_with("pSrc") || name.starts_with("pSource")) && role.contains("ptr") {
            if i + 1 < classified.len() {
                let (next_name, next_role) = &classified[i + 1];
                if (next_name.contains("SrcStep") || next_name.contains("SourceStep"))
                    && next_role.contains("i32")
                {
                    merged.push("SRC+STEP".to_string());
                    i += 2;
                    continue;
                }
            }
        }

        // Check for DST+STEP pattern: pDst followed by nDstStep
        if (name.starts_with("pDst") || name.starts_with("pDestination")) && role.contains("ptr") {
            if i + 1 < classified.len() {
                let (next_name, next_role) = &classified[i + 1];
                if (next_name.contains("DstStep") || next_name.contains("DestinationStep"))
                    && next_role.contains("i32")
                {
                    merged.push("DST+STEP".to_string());
                    i += 2;
                    continue;
                }
            }
        }

        // Check for KERNEL+KSIZE+ANCHOR pattern
        if name.starts_with("pKernel") && role.contains("ptr") {
            // Look ahead for NppiSize (kernel size) and anchor
            let mut j = i + 1;
            let mut has_ksize = false;
            let mut has_anchor = false;
            while j < classified.len() && j <= i + 3 {
                let (n, _) = &classified[j];
                if n.contains("KSize") || n.contains("oKernelSize") {
                    has_ksize = true;
                }
                if n.contains("Anchor") || n.contains("oAnchor") {
                    has_anchor = true;
                }
                j += 1;
            }
            if has_ksize && has_anchor {
                merged.push("KERNEL+KSIZE+ANCHOR".to_string());
                i = j;
                continue;
            }
        }

        // Check for CHANNEL_ORDER: const int[] named aDstOrder / aOrder / aConstants
        if (name.starts_with("aDstOrder") || name.starts_with("aOrder") || name.starts_with("aConstants"))
            && role.contains("ptr")
        {
            merged.push("CHANNEL_ORDER".to_string());
            i += 1;
            continue;
        }

        // Check for CONST_ARRAY: const int[]
        if (name.starts_with("aDstOrder") || name.starts_with("aOrder") || name.starts_with("aConstants"))
            && role == "MISC:i32"
        {
            merged.push("CHANNEL_ORDER".to_string());
            i += 1;
            continue;
        }

        // Check for OUT_SCALAR: pointer to pMean/pMin/pMax/pSum/pStdDev
        if (name.contains("pMean") || name.contains("pMin") || name.contains("pMax")
            || name.contains("pSum") || name.contains("pStdDev"))
            && role.contains("ptr")
        {
            merged.push("OUT_SCALAR".to_string());
            i += 1;
            continue;
        }

        // Check for SCRATCH_BUF: pBuffer/ppBuffer/hpBuffer
        if (name.contains("Buffer") || name.contains("buffer") || name.contains("hpBuffer"))
            && (role.contains("ptr") || role.contains("i32") || role.contains("u32"))
        {
            if role.contains("i32") || role.contains("u32") {
                merged.push("SCRATCH_BUF".to_string());
                i += 1;
                continue;
            }
        }

        // Otherwise, just add the role as-is
        merged.push(role.clone());
        i += 1;
    }

    merged.join(", ")
}

/// Classify a single parameter
fn classify_param(ty: &syn::Type, pat: &syn::Pat) -> String {
    let param_name = extract_param_name(pat);

    match ty {
        syn::Type::Ptr(ptr) => {
            let inner_ty = &ptr.elem;

            // Check for pointer types
            if let syn::Type::Path(type_path) = &**inner_ty {
                let type_name = type_path
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();

                match type_name.as_str() {
                    // Pixel data pointers
                    "Npp8u" | "Npp8s" | "Npp16u" | "Npp16s" | "Npp32u" | "Npp32s" | "Npp32f" | "Npp64f" => {
                        "ptr:pixel".to_string()
                    }
                    // Step parameters (int pointers)
                    "c_int" | "i32" => {
                        if param_name.contains("Step") || param_name.contains("step") {
                            "i32:step".to_string()
                        } else if param_name.contains("Divisor")
                            || param_name.contains("Value")
                            || param_name.contains("Constant")
                            || param_name.contains("ScaleFactor")
                        {
                            "CONST_SCALAR".to_string()
                        } else {
                            "MISC:i32".to_string()
                        }
                    }
                    // Buffer pointers
                    "c_uint" | "u32" | "u8" => {
                        if param_name.contains("Buffer") || param_name.contains("buffer") {
                            "SCRATCH_BUF".to_string()
                        } else if param_name.contains("Divisor")
                            || param_name.contains("Value")
                            || param_name.contains("Constant")
                            || param_name.contains("ScaleFactor")
                        {
                            "CONST_SCALAR".to_string()
                        } else {
                            "MISC:ptr".to_string()
                        }
                    }
                    _ => "MISC:ptr".to_string(),
                }
            } else {
                "MISC:ptr".to_string()
            }
        }
        syn::Type::Path(type_path) => {
            let type_name = type_path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();

            match type_name.as_str() {
                "NppiSize" => "SIZE".to_string(),
                "NppiRect" => "RECT".to_string(),
                "NppiPoint" => {
                    if param_name.contains("anchor") {
                        "ANCHOR".to_string()
                    } else {
                        "POINT".to_string()
                    }
                }
                "NppStreamContext" => "SKIP".to_string(),
                "c_int" | "i32" => {
                    if param_name.contains("Interpolation") {
                        "INTERP".to_string()
                    } else if param_name.contains("Divisor")
                        || param_name.contains("Value")
                        || param_name.contains("Constant")
                        || param_name.contains("ScaleFactor")
                    {
                        "CONST_SCALAR".to_string()
                    } else if param_name.contains("BufferSize") || param_name.contains("bufferSize")
                        || param_name.contains("nBufferSize")
                    {
                        "SCRATCH_BUF".to_string()
                    } else {
                        "MISC:i32".to_string()
                    }
                }
                _ => format!("MISC:{}", type_name),
            }
        }
        _ => "MISC:other".to_string(),
    }
}

/// Extract parameter name from pattern
fn extract_param_name(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
        _ => String::new(),
    }
}
