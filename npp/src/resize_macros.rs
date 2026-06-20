/// Macro: generate `impl Resize for CudaImage<$rust_ty>`.
///
/// # Arguments
///
/// * `$rust_ty` — the Rust pixel element type (e.g. `u8`, `f32`).
/// * `$token` — the NPP type token string (e.g. `"8u"`, `"32f"`).
/// * `{$($ch:literal => $sym:path),+}` — channel-count arms mapping to NPP symbols,
///   e.g. `{ 3 => npp_sys::nppiResize_8u_C3R, 4 => npp_sys::nppiResize_8u_C4R }`.
///
/// # Expansion
///
/// Expands to an `impl Resize for CudaImage<$rust_ty>` block with:
///
/// 1. A runtime mode guard against the GPU-probed `RESIZE_CAPS` table.
/// 2. `NppiSize` / `NppiRect` setup from image dimensions.
/// 3. Byte-step calculation (`height_stride * size_of::<$rust_ty>()`).
/// 4. Raw-pointer extraction via `DevicePtr`/`DevicePtrMut`, offset by `img_index`.
/// 5. `match self.channels()` dispatching to the channel-specific NPP symbol.
/// 6. `check_status(status)` returning `Result<(), NppError>`.
///
/// # Safety
///
/// `self` and `dst` must refer to **non-overlapping** device buffers (C4).
#[macro_export]
macro_rules! impl_resize_for {
    ($rust_ty:ty, $token:expr, { $($ch:literal => $sym:path),+ $(,)? }) => {
        impl Resize for CudaImage<$rust_ty> {
            #[doc = concat!(
                "Resize for `CudaImage<",
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
            /// The raw pointer for both src and dst is offset by `layout.img_index` so
            /// this impl works correctly on sub-images created via `CudaImage::sub_image`
            /// (whose `layout.img_index` carries the parent's offset).
            ///
            /// # Precondition
            ///
            /// `self` and `dst` must refer to **non-overlapping** device buffers.
            /// Passing overlapping ROIs is undefined behaviour in NPP (C4).
            ///
            /// # Errors
            ///
            /// Returns `NppError::InvalidArgument` if the interpolation mode is not
            /// supported for this type (checked against the probed `RESIZE_CAPS` table),
            /// or if `self.channels()` is not one of the supported channel counts.
            /// Returns `NppError::Npp` if the underlying NPP call fails.
            fn resize(
                &self,
                dst: &mut Self,
                inter: ResizeInterpolation,
            ) -> Result<(), NppError> {
                if !$crate::resize_ops::mode_supported($token, inter) {
                    return Err(NppError::InvalidArgument(format!(
                        "Resize mode {inter:?} is not supported for type {type_token}",
                        type_token = $token,
                    )));
                }

                let (src_w, src_h) = (self.width(), self.height());
                let (dst_w, dst_h) = (dst.width(), dst.height());

                let src_size = NppiSize {
                    width: src_w as i32,
                    height: src_h as i32,
                };
                let dst_size = NppiSize {
                    width: dst_w as i32,
                    height: dst_h as i32,
                };
                let src_rect = NppiRect {
                    x: 0,
                    y: 0,
                    width: src_w as i32,
                    height: src_h as i32,
                };
                let dst_rect = NppiRect {
                    x: 0,
                    y: 0,
                    width: dst_w as i32,
                    height: dst_h as i32,
                };

                // nStep is in BYTES. height_stride counts elements. Convert.
                let src_step_bytes =
                    (self.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;
                let dst_step_bytes =
                    (dst.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;

                // ── Raw pointers via DevicePtr/DevicePtrMut (handles sub-images) ──
                let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
                let src_ptr = (src_base + self.layout.img_index as u64) as *const $rust_ty;
                let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
                let dst_ptr = (*dst_base + dst.layout.img_index as u64) as *mut $rust_ty;

                let status = unsafe {
                    match self.channels() {
                        $(
                            $ch => $sym(
                                src_ptr as *const _,
                                src_step_bytes,
                                src_size,
                                src_rect,
                                dst_ptr as *mut _,
                                dst_step_bytes,
                                dst_size,
                                dst_rect,
                                $crate::resize_ops::interpolation_mode(inter),
                            ),
                        )+
                        _ => {
                            return Err(NppError::InvalidArgument(format!(
                                "unsupported channel count {} for Resize with type {}",
                                self.channels(),
                                stringify!($rust_ty),
                            )));
                        }
                    }
                };
                check_status(status)
            }
        }
    };
}
