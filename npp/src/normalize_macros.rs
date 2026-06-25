/// Macro: generate `impl Normalize<f32> for CudaImage<$src_ty>`.
///
/// # Arguments
///
/// * `$src_ty` — the Rust source pixel element type (e.g. `u8`, `u16`, `i16`).
/// * `$denominator` — the f32 literal for the source type's maximum positive
///   representable value (e.g. `255.0_f32` for `u8`, `65535.0_f32` for `u16`).
/// * `$src_token` — the NPP source type token string (e.g. `"8u"`, `"16u"`),
///   used for error messages and doc generation.
///
/// # Expansion
///
/// Expands to an `impl Normalize<f32> for CudaImage<$src_ty>` block with two
/// legs:
///
/// 1. **Convert step** — delegates to the `ConvertTo` trait via
///    `self.convert(dst)?`, which resolves to the generated
///    `ConvertTo<f32> for CudaImage<$src_ty>` in `convert_generated.rs`.
///    This handles dimension validation, pointer extraction, step calculation,
///    sub-image `img_index` offset, and channel dispatch.
///
/// 2. **Scale step** — in-place `MulC` on `dst` with constant
///    `1.0_f32 / $denominator`. The three NPP symbols
///    (`nppiMulC_32f_C1R_Ctx`, `_C3R_Ctx`, `_C4R_Ctx`) are hardcoded in the
///    macro body; the channel count is dispatched at runtime via
///    `match dst.channels()`.
///
/// # In-place aliasing note
///
/// `dst_ptr` is passed as both source and destination to `MulC`. This is sound
/// because `MulC` is purely elementwise — output pixel (x, y) depends only on
/// input pixel (x, y). No gather-overlap hazard (C4).
///
/// # Sub-image support
///
/// NOTE: ROI sub-image support was **implemented in F6.2 for Resize and
/// SwapChannels only**. Normalize remains **owned-buffer only** (`img_index` is
/// always 0 for owned images; the pointer arithmetic does not apply a `img_index`
/// offset).
///
/// # Precondition
///
/// The CUDA device handle stored in `CudaImage` must outlive all buffers
/// created from it (C7).
///
/// `self` and `dst` must not overlap in memory. The convert step obeys the
/// non-overlap requirement for **neighbourhood-gather** operations; the scale
/// step is purely **elementwise** and may safely alias.
///
/// # Errors
///
/// Returns `NppError::InvalidArgument` if `self` and `dst` dimensions or
/// channel counts disagree (from the convert step), or if `dst.channels()` is
/// not one of {1, 3, 4}.
/// Returns `NppError::Npp` if the underlying NPP call fails.
///
/// # Notes
///
/// Normalization of float source types (`f32`, `f64`) is excluded — there is
/// no defined denominator. For signed integer types, negative inputs map below
/// `0.0` (e.g. `-32768 → ~-1.000031` for `i16`).
#[macro_export]
macro_rules! impl_normalize_for {
    ($src_ty:ty, $denominator:expr, $src_token:expr) => {
        /// Normalize this image to `f32` using the generated ConvertTo impl and in-place MulC.
        ///
        /// Dispatches on `dst.channels()` at runtime for C1/C3/C4.
        ///
        /// # Implementation
        ///
        /// 1. Convert via `ConvertTo<f32>` — resolves to the generated
        ///    `impl ConvertTo<f32> for CudaImage<` $src_ty `>` in
        ///    `convert_generated.rs`.
        /// 2. Scale in-place via `nppiMulC_32f_C*R_Ctx` with constant
        ///    `1 /` $denominator `.
        ///
        /// The in-place scaling is safe because `MulC` is a purely elementwise
        /// operation — output pixel (x, y) depends only on input pixel (x, y).
        ///
        /// # Precondition
        ///
        /// `self` and `dst` must not overlap in memory. The convert step obeys
        /// the non-overlap requirement for **neighbourhood-gather** operations;
        /// the scale step is purely **elementwise** and may safely alias.
        ///
        /// The destination image's `width`, `height`, and `channels` must
        /// match the source.
        ///
        /// The CUDA device handle must outlive all buffers created from it
        /// (C7).
        #[allow(clippy::macro_metavars_in_unsafe)]
        impl Normalize<f32> for CudaImage<$src_ty> {
            fn normalize(&self, dst: &mut CudaImage<f32>) -> Result<(), NppError> {
                // Step 1: Convert via ConvertTo trait.
                // This handles dimension validation, pointer extraction, step
                // calculation, sub-image offset, and channel dispatch.
                self.convert(dst)?;

                // Step 2: Scale in-place by 1/denominator.
                // MulC is elementwise — output (x, y) depends only on input
                // (x, y) — so reading/writing dst in place is sound and does
                // not trigger the C4 gather-overlap hazard.

                let scale: f32 = 1.0_f32 / $denominator;

                // SAFETY: The device pointer is valid because dst.buf is a
                // live CudaSlice<f32> allocated from the same device. The
                // img_index offset handles sub-image regions. MulC is
                // elementwise, so aliasing src == dst is safe.
                let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
                let dst_ptr = *dst_base as *mut f32;

                let dst_step = (dst.layout.height_stride * std::mem::size_of::<f32>()) as i32;

                let size = npp_sys::NppiSize {
                    width: dst.width() as i32,
                    height: dst.height() as i32,
                };

                let status = unsafe {
                    match dst.channels() {
                        1 => npp_sys::nppiMulC_32f_C1R_Ctx(
                            dst_ptr,
                            dst_step,
                            scale,
                            dst_ptr,
                            dst_step,
                            size,
                            dst.ctx.raw_ctx(),
                        ),
                        3 => {
                            let constants: [f32; 3] = [scale; 3];
                            npp_sys::nppiMulC_32f_C3R_Ctx(
                                dst_ptr,
                                dst_step,
                                constants.as_ptr(),
                                dst_ptr,
                                dst_step,
                                size,
                                dst.ctx.raw_ctx(),
                            )
                        }
                        4 => {
                            let constants: [f32; 4] = [scale; 4];
                            npp_sys::nppiMulC_32f_C4R_Ctx(
                                dst_ptr,
                                dst_step,
                                constants.as_ptr(),
                                dst_ptr,
                                dst_step,
                                size,
                                dst.ctx.raw_ctx(),
                            )
                        }
                        _ => {
                            return Err(NppError::InvalidArgument(format!(
                                "unsupported channel count {} for normalize {}→f32",
                                dst.channels(),
                                $src_token,
                            )));
                        }
                    }
                };
                check_status(status)?;
                Ok(())
            }
        }
    };
}
