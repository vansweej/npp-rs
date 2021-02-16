mod cuda_resize_image_with_cuda_malloc;
mod cuda_resize_image_with_nppiMalloc;
mod rust_resize_image;

use crate::cuda_resize_image_with_cuda_malloc::*;
use crate::cuda_resize_image_with_nppiMalloc::*;
use crate::rust_resize_image::*;
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    resize_rust_image_crate_benchmark,
    cuda_resize_benchmark_with_nppi_malloc,
    cuda_resize_benchmark_with_cuda_malloc
);
criterion_main!(benches);
