use core::ffi::c_void;
use criterion::Criterion;
use cuda_runtime_sys::{cudaMemcpy, cudaMemcpyKind};
use image::io::Reader as ImageReader;
use npp_sys::{
    nppiFree, nppiMalloc_8u_C3, nppiResize_8u_C3R, NppiInterpolationMode_NPPI_INTER_LINEAR,
    NppiRect, NppiSize,
};

// benchmark with rust image crate
pub fn cuda_resize_benchmark_with_nppi_malloc(c: &mut Criterion) {
    let mut src_stride: i32 = 0;
    let mut dst_stride: i32 = 0;

    let img = ImageReader::open("test_resources/DSC_0003.JPG")
        .unwrap()
        .decode()
        .unwrap();
    let img_layout = img.as_rgb8().unwrap().sample_layout();

    //allocate cuda memory for the images
    let cuda_src = unsafe {
        nppiMalloc_8u_C3(
            img_layout.width as i32,
            img_layout.height as i32,
            &mut src_stride,
        )
    };

    let cuda_dst = unsafe { nppiMalloc_8u_C3(640, 480, &mut dst_stride) };

    let img_raw_samples = img.as_rgb8().unwrap().as_flat_samples();
    for h in 0..img_layout.height {
        let begin_row = h as usize * img_layout.height_stride;
        let end_row = begin_row + img_layout.height_stride as usize - 1;

        let _err = unsafe {
            cudaMemcpy(
                cuda_src.offset(begin_row as isize) as *mut c_void,
                img_raw_samples.as_slice()[begin_row..end_row].as_ptr() as *const c_void,
                img_layout.height_stride,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
    }

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

    c.bench_function("resize with cuda with nppi malloc", |b| {
        b.iter(|| {
            let _status = unsafe {
                nppiResize_8u_C3R(
                    cuda_src,
                    src_stride,
                    src_size,
                    src_rect,
                    cuda_dst,
                    dst_stride,
                    dst_size,
                    dst_rect,
                    NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
                )
            };
        })
    });

    unsafe {
        nppiFree(cuda_dst as *mut c_void);
        nppiFree(cuda_src as *mut c_void);
    }
}
