/// Macro: generate a `pub(crate)` ROI engine function + `impl SwapChannels for CudaImage<$rust_ty>`.
///
/// # Arguments
///
/// * `$fn_name` — name of the emitted free `pub(crate)` engine function (e.g. `swap_into_8u`).
/// * `$rust_ty` — the Rust pixel element type (e.g. `u8`, `f32`).
/// * `$token` — the NPP type token string (e.g. `"8u"`, `"32f"`).
/// * `{$($ch:literal => $sym:path),+}` — channel-count arms mapping to NPP symbols,
///   e.g. `{ 4 => npp_sys::nppiSwapChannels_8u_C4C3R }`.
///
/// # Expansion
///
/// 1. A free `pub(crate)` engine function (`$fn_name`) with the raw-pointer FFI call
///    sequence. Takes all arguments explicitly (no `CudaImage` reference) so it can
///    serve both owned `CudaImage` calls and borrowed `CudaImageView` sub-image calls.
/// 2. An `impl SwapChannels for CudaImage<$rust_ty>` block whose `fn bgra_to_rgb` body
///    is a thin wrapper over the engine function.
///
/// # Engine function arguments
///
/// - `src_ptr`, `src_step_bytes`, `src_w`, `src_h`, `src_channels` — source parameters
/// - `dst_ptr`, `dst_step_bytes`, `dst_w`, `dst_h` — destination parameters
/// - `ctx` — raw `NppStreamContext` (by value, `Copy`)
///
/// # Safety
///
/// `src_ptr` and `dst_ptr` must not overlap in memory. This applies to
/// **neighbourhood-gather** operations; aliasing produces undefined results.
/// Purely **elementwise** operations may safely alias (see `Normalize`).
#[macro_export]
macro_rules! impl_swap_channels_for {
    ($fn_name:ident, $rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
        /// ROI swap channels engine: reorder BGRA→RGB using raw device pointers.
        ///
        /// This `pub(crate)` function accepts explicit pointers and dimensions so it can
        /// be called from both the owned `SwapChannels` trait impl and from
        /// `CudaImageView`-based sub-image paths. The FFI call sequence is identical
        /// to the owned path.
        ///
        /// # nStep unit conversion
        ///
        /// NPP's `nStep` is in **bytes**. Callers must pass `height_stride * size_of::<T>()`.
        ///
        /// # Precondition
        ///
        /// `src_ptr` and `dst_ptr` must not overlap in memory. This applies to
        /// **neighbourhood-gather** operations; aliasing produces undefined results.
        /// Purely **elementwise** operations may safely alias.
        ///
        /// `src_w` / `src_h` must equal `dst_w` / `dst_h` — SwapChannels does not
        /// resize, it only reorders channels.
        ///
        /// # Errors
        ///
        /// Returns `NppError::InvalidArgument` if src and dst dimensions disagree,
        /// or if `src_channels` is not one of the supported channel counts.
        /// Returns `NppError::Npp` if the underlying NPP call fails.
        #[allow(clippy::too_many_arguments)]
        #[cfg(not(tarpaulin_include))]
        pub(crate) fn $fn_name(
            src_ptr: *const $rust_ty,
            src_step_bytes: i32,
            src_w: u32,
            src_h: u32,
            src_channels: u8,
            dst_ptr: *mut $rust_ty,
            dst_step_bytes: i32,
            dst_w: u32,
            dst_h: u32,
            ctx: npp_sys::NppStreamContext,
        ) -> Result<(), NppError> {
            if src_w != dst_w || src_h != dst_h {
                return Err(NppError::InvalidArgument(
                    "src and dst dimensions must match for bgra_to_rgb".into(),
                ));
            }
            let nppi_size = NppiSize { width: dst_w as i32, height: dst_h as i32 };
            let order: [std::os::raw::c_int; 3] = [2, 1, 0];
            let status = unsafe {
                match src_channels {
                    $(
                        $ch => $sym(
                            src_ptr as *const _,
                            src_step_bytes,
                            dst_ptr as *mut _,
                            dst_step_bytes,
                            nppi_size,
                            &order[0],
                            ctx,
                        ),
                    )+
                    _ => {
                        return Err(NppError::InvalidArgument(format!(
                            "unsupported channel count {} for SwapChannels with type {}",
                            src_channels,
                            stringify!($rust_ty),
                        )));
                    }
                }
            };
            check_status(status)
        }

        impl SwapChannels for CudaImage<$rust_ty> {
            #[doc = concat!(
                "BGRA→RGB reorder for `CudaImage<",
                stringify!($rust_ty),
                ">` over NPP type token `",
                $token,
                "`. Dispatches on `self.channels()` at runtime.",
            )]
            ///
            /// # nStep unit conversion
            ///
            /// NPP's `nStep` is in **bytes**. `layout.height_stride` stores the per-row
            /// element count; we multiply by `size_of::<T>()` to produce the byte step.
            ///
            /// This is a thin wrapper over the [`swap_into_`] engine function, which
            /// also serves ROI sub-image views via `CudaImageView::device_ptr()`.
            ///
            /// # Precondition
            ///
            /// `self` and `dst` must not overlap in memory. This applies to
            /// **neighbourhood-gather** operations; aliasing produces undefined results.
            /// Purely **elementwise** operations may safely alias (see `Normalize`).
            ///
            /// # Errors
            ///
            /// Returns `NppError::InvalidArgument` if src and dst dimensions disagree,
            /// or if `self.channels()` is not one of the supported channel counts.
            /// Returns `NppError::Npp` if the underlying NPP call fails.
            fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError> {
                let src_step_bytes =
                    (self.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
                let dst_step_bytes =
                    (dst.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
                let src_ptr =
                    *cudarc::driver::DevicePtr::device_ptr(&self.buf) as *const $rust_ty;
                let dst_ptr =
                    *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut $rust_ty;
                $fn_name(
                    src_ptr, src_step_bytes, self.width(), self.height(), self.channels(),
                    dst_ptr, dst_step_bytes, dst.width(), dst.height(),
                    self.ctx.raw_ctx(),
                )
            }
        }
    };
}
