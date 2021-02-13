mod rust_image;
use crate::rust_image::*;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, resize_rust_image_crate_benchmark);
criterion_main!(benches);
