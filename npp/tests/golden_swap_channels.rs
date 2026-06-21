//! Golden-image correctness test for `SwapChannels` on `CudaImage<u8>`.
//!
//! Proves that the macro-generated `bgra_to_rgb` produces correct pixels,
//! not just correct geometry (C12 minimum mitigation).
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_swap_channels`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.

#![cfg(feature = "gpu")]

use cudarc::driver::CudaDevice;
use npp_rs::cuda::default_cuda_device;
use npp_rs::image::CudaImage;
use npp_rs::imageops::SwapChannels;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;
use std::sync::Arc;

const W: u32 = 12;
const H: u32 = 8;

/// Input: procedurally generated 4-channel BGRA gradient (12x8).
///
/// Channel layout per pixel: [B, G, R, A]
fn make_input() -> Vec<u8> {
    let mut data = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            data.push((x * 21) as u8); // B: x-gradient
            data.push((y * 32) as u8); // G: y-gradient
            data.push(128); // R: constant
            data.push(255); // A: full opacity
        }
    }
    data
}

/// Golden output for BGRA→RGB (B and R swapped, A dropped).
/// Generated on NVIDIA GPU. Bit-exact (no floating point).
///
/// NOTE: Set this to `&[]` initially, run on GPU, copy the printed bytes here.
const EXPECTED: &[u8] = &[
    // Golden data will be pinned after first GPU test run
];

#[test]
fn test_golden_swap_channels_u8() {
    let device: Arc<CudaDevice> = default_cuda_device().expect("CUDA device init");

    // 4-channel source (BGRA)
    let src =
        CudaImage::from_host(device.clone(), 4, W, H, &make_input()).expect("src allocation");

    // 3-channel destination (RGB)
    let mut dst = CudaImage::<u8>::new(device.clone(), 3, W, H).expect("dst allocation");

    src.bgra_to_rgb(&mut dst).expect("bgra_to_rgb");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED, "bgra_to_rgb");
}
