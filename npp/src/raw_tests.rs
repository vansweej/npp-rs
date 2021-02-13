#[cfg(test)]
mod tests {
    //    use super::*;
    use core::ffi::c_void;
    use npp_sys::{
        nppiFree, nppiMalloc_8u_C1, nppiMalloc_8u_C2, nppiMalloc_8u_C3, nppiMalloc_8u_C4,
    };

    #[test]
    fn test_allocations() {
        let width = 640;
        let height = 480;
        let mut stride: i32 = 0;

        unsafe {
            let x = nppiMalloc_8u_C1(width, height, &mut stride);

            assert_eq!(stride, 1024);

            nppiFree(x as *mut c_void);
        }
        unsafe {
            let x = nppiMalloc_8u_C2(width, height, &mut stride);

            assert_eq!(stride, 1536);

            nppiFree(x as *mut c_void);
        }
        unsafe {
            let x = nppiMalloc_8u_C3(width, height, &mut stride);

            assert_eq!(stride, 2048);

            nppiFree(x as *mut c_void);
        }
        unsafe {
            let x = nppiMalloc_8u_C4(width, height, &mut stride);

            assert_eq!(stride, 2560);

            nppiFree(x as *mut c_void);
        }
    }
}
