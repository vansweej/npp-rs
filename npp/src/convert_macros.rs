/// Macro: generate `impl ConvertTo<$dst_ty> for CudaImage<$src_ty>`.
///
/// # Arguments
///
/// * `$src_ty` — the Rust source pixel element type (e.g. `u8`, `u16`).
/// * `$dst_ty` — the Rust destination pixel element type (e.g. `f32`, `u16`).
/// * `$src_token` — the NPP source type token string (e.g. `"8u"`, `"16u"`).
/// * `$dst_token` — the NPP destination type token string (e.g. `"32f"`, `"16u"`).
/// * `{$($ch:literal => $sym:path),+}` — channel-count arms mapping to NPP symbols,
///   e.g. `{ 1 => npp_sys::nppiConvert_8u32f_C1R_Ctx, 3 => npp_sys::nppiConvert_8u32f_C3R_Ctx }`.
///
/// # Expansion
///
/// Expands to an `impl ConvertTo<$dst_ty> for CudaImage<$src_ty>` block with:
///
/// 1. Dimension/channel agreement check (src vs dst).
/// 2. `NppiSize` setup from destination dimensions.
/// 3. Byte-step calculation: `height_stride * size_of::<$src_ty>()` for src,
///    `height_stride * size_of::<$dst_ty>()` for dst (the differing element sizes
///    are the crux of cross-type stepping).
/// 4. Raw-pointer extraction via `DevicePtr`/`DevicePtrMut`, offset by `img_index`
///    for sub-image support.
/// 5. `match self.channels()` dispatching to the channel-specific NPP symbol.
/// 6. `check_status(status)` returning `Result<(), NppError>`.
///
/// This implements **unscaled, non-rounding** conversion. Scaled/rounding
/// conversion (via `NppRoundMode`) is **F5.2**.
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
macro_rules! impl_convert_for {
    ($src_ty:ty, $dst_ty:ty, $src_token:expr, $dst_token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
        #[doc = concat!(
            "Convert `CudaImage<",
            stringify!($src_ty),
            ">` to `CudaImage<",
            stringify!($dst_ty),
            ">` over NPP type tokens `",
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
            /// NOTE: ROI sub-image support was **implemented in F6.2 for Resize and
            /// SwapChannels only**. Convert remains **owned-buffer only** (`img_index` is
            /// always 0 for owned images; the pointer arithmetic does not apply a `img_index`
            /// offset).
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
        /// supported channel counts.
        /// Returns `NppError::Npp` if the underlying NPP call fails.
        #[allow(clippy::macro_metavars_in_unsafe)]
        impl ConvertTo<$dst_ty> for CudaImage<$src_ty> {
            fn convert(&self, dst: &mut CudaImage<$dst_ty>) -> Result<(), NppError> {
                // Check 1: Agreement — dimensions and channels must match
                if self.width() != dst.width()
                    || self.height() != dst.height()
                    || self.channels() != dst.channels()
                {
                    return Err(NppError::InvalidArgument(
                        "src and dst dimensions/channels must match for convert".into(),
                    ));
                }

                // nStep is in BYTES. height_stride counts elements. Convert.
                // src and dst have potentially different element sizes.
                let src_step_bytes =
                    (self.layout.height_stride * std::mem::size_of::<$src_ty>()) as i32;
                let dst_step_bytes =
                    (dst.layout.height_stride * std::mem::size_of::<$dst_ty>()) as i32;

                // ── Raw pointers via DevicePtr/DevicePtrMut ──
                let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
                let src_ptr = *src_base as *const $src_ty;

                let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
                let dst_ptr = *dst_base as *mut $dst_ty;

                let nppi_size = NppiSize {
                    width: dst.width() as i32,
                    height: dst.height() as i32,
                };

                let status = unsafe {
                    match self.channels() {
                        $(
                            $ch => $sym(
                                src_ptr as *const _,
                                src_step_bytes,
                                dst_ptr as *mut _,
                                dst_step_bytes,
                                nppi_size,
                                self.ctx.raw_ctx(),
                            ),
                        )+
                        _ => {
                            return Err(NppError::InvalidArgument(format!(
                                "unsupported channel count {} for convert {} → {}",
                                self.channels(),
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
