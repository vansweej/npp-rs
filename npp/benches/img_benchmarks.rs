mod cuda_alloc_and_copy_img_to_device_memory;
mod cuda_resize_image_with_cuda_malloc;
mod cuda_resize_image_with_nppi_malloc;
mod rust_resize_image;

use crate::cuda_alloc_and_copy_img_to_device_memory::*;
use crate::cuda_resize_image_with_cuda_malloc::*;
use crate::cuda_resize_image_with_nppi_malloc::*;
use crate::rust_resize_image::*;
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    resize_rust_image_crate_benchmark,
    cuda_resize_benchmark_with_nppi_malloc,
    cuda_resize_benchmark_with_cuda_malloc,
    cuda_alloc_and_copy_img_to_device_memory_benchmark
);
criterion_main!(benches);
