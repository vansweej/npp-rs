use criterion::Criterion;
use image::io::Reader as ImageReader;
use image::*;

// benchmark with rust image crate
pub fn resize_rust_image_crate_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("test_resources/DSC_0003.JPG")
        .unwrap()
        .decode()
        .unwrap();

    c.bench_function("resize with rust image crate", |b| {
        b.iter(|| {
            let _resized_img = imageops::resize(&img, 640, 480, imageops::FilterType::Nearest);
        })
    });
}
