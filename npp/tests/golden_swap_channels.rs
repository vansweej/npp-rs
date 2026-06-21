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
    128, 0, 0, 128, 0, 21, 128, 0, 42, 128, 0, 63, 128, 0, 84, 128, 0, 105, 128, 0, 126, 128, 0,
    147, 128, 0, 168, 128, 0, 189, 128, 0, 210, 128, 0, 231, 128, 32, 0, 128, 32, 21, 128, 32, 42,
    128, 32, 63, 128, 32, 84, 128, 32, 105, 128, 32, 126, 128, 32, 147, 128, 32, 168, 128, 32, 189,
    128, 32, 210, 128, 32, 231, 128, 64, 0, 128, 64, 21, 128, 64, 42, 128, 64, 63, 128, 64, 84,
    128, 64, 105, 128, 64, 126, 128, 64, 147, 128, 64, 168, 128, 64, 189, 128, 64, 210, 128, 64,
    231, 128, 96, 0, 128, 96, 21, 128, 96, 42, 128, 96, 63, 128, 96, 84, 128, 96, 105, 128, 96,
    126, 128, 96, 147, 128, 96, 168, 128, 96, 189, 128, 96, 210, 128, 96, 231, 128, 128, 0, 128,
    128, 21, 128, 128, 42, 128, 128, 63, 128, 128, 84, 128, 128, 105, 128, 128, 126, 128, 128, 147,
    128, 128, 168, 128, 128, 189, 128, 128, 210, 128, 128, 231, 128, 160, 0, 128, 160, 21, 128,
    160, 42, 128, 160, 63, 128, 160, 84, 128, 160, 105, 128, 160, 126, 128, 160, 147, 128, 160,
    168, 128, 160, 189, 128, 160, 210, 128, 160, 231, 128, 192, 0, 128, 192, 21, 128, 192, 42, 128,
    192, 63, 128, 192, 84, 128, 192, 105, 128, 192, 126, 128, 192, 147, 128, 192, 168, 128, 192,
    189, 128, 192, 210, 128, 192, 231, 128, 224, 0, 128, 224, 21, 128, 224, 42, 128, 224, 63, 128,
    224, 84, 128, 224, 105, 128, 224, 126, 128, 224, 147, 128, 224, 168, 128, 224, 189, 128, 224,
    210, 128, 224, 231,
];

#[test]
fn test_golden_swap_channels_u8() {
    let device: Arc<CudaDevice> = default_cuda_device().expect("CUDA device init");

    // 4-channel source (BGRA)
    let src = CudaImage::from_host(device.clone(), 4, W, H, &make_input()).expect("src allocation");

    // 3-channel destination (RGB)
    let mut dst = CudaImage::<u8>::new(device.clone(), 3, W, H).expect("dst allocation");

    src.bgra_to_rgb(&mut dst).expect("bgra_to_rgb");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED, "bgra_to_rgb");
}
