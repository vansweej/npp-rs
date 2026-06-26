/// Macro: generate `impl ConvertRoundedScaled<$dst_ty> for CudaImage<$src_ty>`.
///
/// # Arguments
///
/// * `$src_ty` — the Rust source pixel element type (e.g. `f32`).
/// * `$dst_ty` — the Rust destination pixel element type (e.g. `u8`).
/// * `$src_token` — the NPP source type token string (e.g. `"32f"`, `"16u"`).
/// * `$dst_token` — the NPP destination type token string (e.g. `"8u"`, `"8s"`).
/// * `{$($ch:literal => $sym:path),+}` — channel-count arms mapping to NPP symbols,
///   e.g. `{ 1 => npp_sys::nppiConvert_32f8u_C1RSfs_Ctx }`.
///   Only `C1RSfs` symbols exist (single-channel); multi-channel arms are a
///   compile-time error.
///
/// # Expansion
///
/// Expands to an `impl ConvertRoundedScaled<$dst_ty> for CudaImage<$src_ty>` block with:
///
/// 1. Dimension/channel agreement check (src vs dst).
/// 2. Single-channel guard (`self.channels() != 1` → error).
/// 3. `NppiSize` setup from destination dimensions.
/// 4. Byte-step calculation: `height_stride * size_of::<$src_ty>()` for src,
///    `height_stride * size_of::<$dst_ty>()` for dst (the differing element sizes
///    are the crux of cross-type stepping).
/// 5. Raw-pointer extraction via `DevicePtr`/`DevicePtrMut`.
/// 6. `match self.channels()` dispatching to the channel-specific NPP symbol,
///    passing round-mode and scale-factor parameters between `nppi_size` and the
///    stream context.
/// 7. `check_status(status)` returning `Result<(), NppError>`.
///
/// This implements **scaled rounding-mode** conversion (narrowing, e.g. `f32 → u8`)
/// with an additional integer `scale_factor` that NPP applies as a power-of-two
/// exponent before saturating the result (per NPP `Sfs` convention).
///
/// # Safety
///
/// `self` and `dst` must not overlap in memory. This applies to
/// **neighbourhood-gather** operations; aliasing produces undefined results.
/// Purely **elementwise** operations may safely alias (see `Normalize`).
///
/// The CUDA device handle stored in `CudaImage` must outlive all operations
/// (C7).
#[macro_export]
macro_rules! impl_convert_rounded_scaled_for {
    ($src_ty:ty, $dst_ty:ty, $src_token:expr, $dst_token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
        #[doc = concat!(
            "Convert `CudaImage<",
            stringify!($src_ty),
            ">` to `CudaImage<",
            stringify!($dst_ty),
            ">` with explicit rounding mode and scaling factor over NPP type tokens `",
            $src_token,
            "` → `",
            $dst_token,
            "`. Dispatches on `self.channels()` at runtime.",
        )]
        ///
        /// # nStep unit conversion
        ///
        /// NPP's `nStep` is in **bytes**. `layout.height_stride` stores the per-row
        /// element count; we multiply by `size_of::<T>()` to produce the byte step.
        /// Because src and dst have different element types, each has its own
        /// step calculation.
        ///
        /// This is a **narrowing** conversion (src bytes > dst bytes), opposite
        /// of `ConvertTo`'s widening conversions. The `round_mode()` helper
        /// translates the safe `RoundMode` enum to the raw NPP constant.
        ///
        /// The `scale_factor` is applied by NPP as an integer scaling factor
        /// (power-of-two shift per `Sfs` convention) before rounding and
        /// saturation.
        ///
        /// NOTE: This is a **C1-only** operation — NPP does not expose C3/C4
        /// scaled rounding-mode convert symbols.
        ///
        /// # Precondition
        ///
        /// `self` and `dst` must not overlap in memory. This applies to
        /// **neighbourhood-gather** operations; aliasing produces undefined results.
        /// Purely **elementwise** operations may safely alias (see `Normalize`).
        ///
        /// # Errors
        ///
        /// Returns `NppError::InvalidArgument` if `self` and `dst` dimensions or
        /// channel counts disagree, or if `self.channels()` is not one of the
        /// supported channel counts (only `C1` = 1).
        /// Returns `NppError::Npp` if the underlying NPP call fails (including
        /// `NPP_ROUND_MODE_NOT_SUPPORTED_ERROR` if the pair rejects `mode`).
        #[allow(clippy::macro_metavars_in_unsafe)]
        impl ConvertRoundedScaled<$dst_ty> for CudaImage<$src_ty> {
            fn convert_rounded_scaled(
                &self,
                dst: &mut CudaImage<$dst_ty>,
                mode: RoundMode,
                scale_factor: i32,
            ) -> Result<(), NppError> {
                // Check 1: Agreement — dimensions and channels must match
                if self.width() != dst.width()
                    || self.height() != dst.height()
                    || self.channels() != dst.channels()
                {
                    return Err(NppError::InvalidArgument(
                        "src and dst dimensions/channels must match for convert_rounded_scaled"
                            .into(),
                    ));
                }

                // Check 2: Single-channel only (C1RSfs)
                if self.channels() != 1 {
                    return Err(NppError::InvalidArgument(
                        "convert_rounded_scaled is C1-only; multi-channel not supported by NPP"
                            .into(),
                    ));
                }

                // nStep is in BYTES. height_stride counts elements. Convert.
                // src and dst have potentially different element sizes.
                let src_step_bytes =
                    (self.layout.height_stride * std::mem::size_of::<$src_ty>()) as i32;
                let dst_step_bytes =
                    (dst.layout.height_stride * std::mem::size_of::<$dst_ty>()) as i32;

                // Precompute all values before extracting device pointers (avoids
                // E0502 borrow conflict — SyncOnDrop keeps the &mut dst.buf borrow alive).
                let ch = self.channels();
                let nppi_size = NppiSize {
                    width: dst.width() as i32,
                    height: dst.height() as i32,
                };
                let raw_ctx = self.ctx.raw_ctx();

                // ── Raw pointers via DevicePtr/DevicePtrMut ──
                let (src_cu_ptr, _src_guard) =
                    cudarc::driver::DevicePtr::device_ptr(&self.buf, self.ctx.stream());
                let src_ptr = src_cu_ptr as *const $src_ty;

                let (dst_cu_ptr, _dst_guard) =
                    cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, dst.ctx.stream());
                let dst_ptr = dst_cu_ptr as *mut $dst_ty;

                let status = unsafe {
                    match ch {
                        $(
                            $ch => $sym(
                                src_ptr as *const _,
                                src_step_bytes,
                                dst_ptr as *mut _,
                                dst_step_bytes,
                                nppi_size,
                                $crate::convert_round_ops::round_mode(mode),
                                scale_factor,
                                raw_ctx,
                            ),
                        )+
                        _ => {
                            return Err(NppError::InvalidArgument(format!(
                                "unsupported channel count {} for convert_rounded_scaled {} → {}",
                                ch,
                                $src_token,
                                $dst_token,
                            )));
                        }
                    }
                };
                check_status(status)
            }
        }
    };
}
