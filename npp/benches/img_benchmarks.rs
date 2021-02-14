mod cuda_resize_benchmark;
mod rust_image;

use crate::cuda_resize_benchmark::*;
use crate::rust_image::*;
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    resize_rust_image_crate_benchmark,
    cuda_resize_benchmark
);
criterion_main!(benches);
