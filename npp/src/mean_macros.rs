/// Macro: generate `impl Mean for CudaImage<$rust_ty>`.
///
/// # Arguments
///
/// * `$rust_ty` — the Rust pixel element type (e.g. `u8`, `f32`).
/// * `$token` — the NPP type token string (e.g. `"8u"`, `"32f"`).
/// * `{$($ch:literal => ($mean_sym:path, $buffer_sym:path)),+}` — channel-count
///   arms mapping to `(nppiMean_*, nppiMeanGetBufferHostSize_*)` pairs.
///
/// # Expansion
///
/// Expands to an `impl Mean for CudaImage<$rust_ty>` block implementing the
/// NPP two-call dance:
///
/// 1. Query scratch buffer size via `{buffer_sym}`.
/// 2. Allocate scratch buffer on device.
/// 3. Allocate output buffer (`CudaSlice<f64>` × channel count).
/// 4. Call `{mean_sym}` with scratch and output buffers.
/// 5. Read back the per-channel means as `Vec<f64>`.
///
/// # Safety
///
/// The CUDA device handle stored in `CudaImage` must outlive all operations
/// (C7).
#[macro_export]
macro_rules! impl_mean_for {
    ($rust_ty:ty, $token:expr, { $($ch:literal => ($mean_sym:path, $buffer_sym:path)),+ $(,)? }) => {
        impl Mean for CudaImage<$rust_ty> {
            #[doc = concat!(
                "Compute per-channel mean for `CudaImage<",
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
            /// The raw pointer for src is offset by `layout.img_index` so this impl
            /// works correctly on sub-images.
            ///
            /// # Errors
            ///
            /// Returns `NppError::InvalidArgument` if `self.channels()` is not one of
            /// the supported channel counts. Returns `NppError::Npp` if the underlying
            /// NPP call fails.
            /// Returns `NppError::Cuda` if scratch-buffer or output-buffer allocation
            /// fails.
            fn mean(&self) -> Result<Vec<f64>, NppError> {
                let nppi_size = NppiSize {
                    width: self.width() as i32,
                    height: self.height() as i32,
                };

                let ch = self.channels();

                // ── Step 1: query scratch buffer size ──
                let mut buffer_size: usize = 0;
                let status = unsafe {
                    match ch {
                        $(
                            $ch => $buffer_sym(nppi_size, &mut buffer_size as *mut usize),
                        )+
                        _ => {
                            return Err(NppError::InvalidArgument(format!(
                                "unsupported channel count {} for Mean with type {}",
                                ch,
                                stringify!($rust_ty),
                            )));
                        }
                    }
                };
                check_status(status)?;

                // ── Step 2: allocate scratch buffer on device ──
                let mut scratch_buf: cudarc::driver::CudaSlice<u8> =
                    self.device.alloc_zeros::<u8>(buffer_size)?;

                // ── Step 3: allocate output buffer on device ──
                let mut out_buf: cudarc::driver::CudaSlice<f64> =
                    self.device.alloc_zeros::<f64>(ch as usize)?;

                // nStep is in BYTES. height_stride counts elements. Convert.
                let src_step_bytes =
                    (self.layout.height_stride * std::mem::size_of::<$rust_ty>()) as i32;

                // ── Raw pointers via DevicePtr/DevicePtrMut ──
                let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
                let src_ptr = (src_base + self.layout.img_index as u64) as *const $rust_ty;
                let scratch_ptr = {
                    let base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut scratch_buf);
                    *base
                };
                let out_ptr = {
                    let base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut out_buf);
                    *base
                };

                // ── Step 4: compute mean ──
                let status = unsafe {
                    match ch {
                        $(
                            $ch => $mean_sym(
                                src_ptr as *const _,
                                src_step_bytes,
                                nppi_size,
                                scratch_ptr as *mut u8,
                                out_ptr as *mut f64,
                            ),
                        )+
                        _ => unreachable!(), // already handled above
                    }
                };
                check_status(status)?;

                // ── Step 5: read back results ──
                let result: Vec<f64> = self.device.dtoh_sync_copy(&out_buf)?;
                Ok(result)
            }
        }
    };
}
