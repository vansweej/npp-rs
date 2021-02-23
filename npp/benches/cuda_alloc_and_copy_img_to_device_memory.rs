use criterion::Criterion;
use image::io::Reader as ImageReader;
use npp_rs::image::*;
use rustacuda::prelude::*;

pub fn cuda_alloc_and_copy_img_to_device_memory_benchmark(c: &mut Criterion) {
    rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
        .unwrap();
    let img = ImageReader::open("test_resources/DSC_0003.JPG")
        .unwrap()
        .decode()
        .unwrap();

    c.bench_function("cuda_alloc_and_copy_img_to_device memory", |b| {
        b.iter(|| {
            let _cuda_buf = CudaImage::from(&img);
        })
    });
}
