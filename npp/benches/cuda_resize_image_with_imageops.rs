use criterion::Criterion;
use image::io::Reader as ImageReader;
use image::ColorType;
use npp_rs::image::*;
use npp_rs::imageops::*;
use rustacuda::error::CudaError;
use rustacuda::prelude::*;
use std::convert::TryFrom;

// benchmark with rust image crate
pub fn cuda_resize_image_with_imageops(c: &mut Criterion) {
    rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
        .unwrap();
    let img_src = ImageReader::open("test_resources/DSC_0003.JPG")
        .unwrap()
        .decode()
        .unwrap();
    let img_layout = img_src.as_rgb8().unwrap().sample_layout();

    let cuda_src = CudaImage::try_from(img_src.as_rgb8().unwrap()).unwrap();

    let mut cuda_dst = match img_layout.channels {
        3 => CudaImage::new(640, 480, ColorType::Rgb8),
        _ => Err(CudaError::UnknownError),
    }
    .unwrap();

    c.bench_function("cuda resize image with imageops", |b| {
        b.iter(|| {
            let _res = resize(&cuda_src, &mut cuda_dst).unwrap();
        })
    });
}
