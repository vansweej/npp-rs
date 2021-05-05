use rustacuda::error::CudaError;
use rustacuda::prelude::*;

pub fn initialize_cuda_device() -> Result<Context, CudaError> {
    rustacuda::init(rustacuda::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_initialize() {
        let res = initialize_cuda_device();
        assert!(res.is_ok());
    }
}
