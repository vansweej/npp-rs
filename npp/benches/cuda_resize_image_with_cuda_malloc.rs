use core::ffi::c_void;
use criterion::Criterion;
use cuda_runtime_sys::{cudaFree, cudaMalloc, cudaMemcpy2D, cudaMemcpyKind};
use image::io::Reader as ImageReader;
use npp_sys::{nppiResize_8u_C3R, NppiInterpolationMode_NPPI_INTER_LINEAR, NppiRect, NppiSize};
use std::ptr::null_mut;

// benchmark with rust image crate
pub fn cuda_resize_benchmark_with_cuda_malloc(c: &mut Criterion) {
    let mut cuda_src: *mut c_void = null_mut();
    let mut cuda_dst: *mut c_void = null_mut();

    let img = ImageReader::open("test_resources/DSC_0003.JPG")
        .unwrap()
        .decode()
        .unwrap();
    let img_layout = img.as_rgb8().unwrap().sample_layout();

    let _rs_src = unsafe {
        cudaMalloc(
            &mut cuda_src,
            (img_layout.width * img_layout.height * 3) as usize,
        )
    };
    let _rs_dst = unsafe { cudaMalloc(&mut cuda_dst, 640 * 480 * 3) };

    let img_raw_samples = img.as_rgb8().unwrap().as_flat_samples();

    let _cpy_res = unsafe {
        cudaMemcpy2D(
            cuda_src as *mut c_void,
            img_layout.height_stride as usize,
            img_raw_samples.as_slice()[..].as_ptr() as *mut c_void,
            img_layout.width as usize,
            img_layout.width as usize,
            img_layout.height as usize,
            cudaMemcpyKind::cudaMemcpyDefault,
        )
    };

    let src_size: NppiSize = NppiSize {
        width: img_layout.width as i32,
        height: img_layout.height as i32,
    };
    let dst_size: NppiSize = NppiSize {
        width: 640,
        height: 480,
    };

    let src_rect: NppiRect = NppiRect {
        x: 0,
        y: 0,
        width: img_layout.width as i32,
        height: img_layout.height as i32,
    };
    let dst_rect: NppiRect = NppiRect {
        x: 0,
        y: 0,
        width: 640,
        height: 480,
    };

    c.bench_function("resize with cuda with cuda malloc", |b| {
        b.iter(|| {
            let _status = unsafe {
                nppiResize_8u_C3R(
                    cuda_src as *mut u8,
                    img_layout.width as i32 * 3,
                    src_size,
                    src_rect,
                    cuda_dst as *mut u8,
                    640 * 3,
                    dst_size,
                    dst_rect,
                    NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
                )
            };
        })
    });

    unsafe {
        cudaFree(cuda_dst);
        cudaFree(cuda_src);
    }
}
