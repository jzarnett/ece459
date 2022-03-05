#[macro_use]
extern crate rustacuda;
extern crate rustacuda_derive;
extern crate rustacuda_core;

use cuda_sys::vector_types::{float1};
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use rand::Rng;
use rustacuda_core::DeviceCopy;
use std::ops::Deref;

/* A Rustification by Jeff Zarnett of a past ECE 459 N-Body assignment that was
originally from GPU Gems, Chapter 31 and modified by Patrick Lam.
Then CUDA-fied by Jeff Zarnett using the Rustacuda library example code
 */

const NUM_POINTS: u32 = 100;
const SPACE: f32 = 1000.0;

struct CudaFloat1(float1);
unsafe impl DeviceCopy for CudaFloat1 {}
impl Deref for CudaFloat1 {
    type Target = float1;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let A = create_input();
    println! {"A:"}
    for pt in A.iter() {
        println! {"({})", pt.x};
    }

    let B = create_input();
    println! {"B:"}
    for pt in B.iter() {
        println! {"({})", pt.x};
    }

    let ptx = CString::new(include_str!("../resources/vector_add.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut A_buf = DeviceBuffer::from_slice(A.as_slice())?;
    let mut B_buf = DeviceBuffer::from_slice(B.as_slice())?;
    let mut C_buf = DeviceBuffer::from_slice(&[0.0f32; NUM_POINTS as usize])?;

    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.vector_add<<<NUM_POINTS, 1, 0, stream>>>(
            A_buf.as_device_ptr(),
            B_buf.as_device_ptr(),
            C_buf.as_device_ptr()
        ));
        result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    let mut C = [0.0f32; NUM_POINTS as usize];
    // Copy the results back to host memory
    C_buf.copy_to(&mut C)?;

    println! {"C:"}
    for pt in C.iter() {
        println! {"({})", pt};
    }
    Ok(())
}

fn create_input() -> Vec<CudaFloat1> {
    let mut result: Vec<CudaFloat1> = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..NUM_POINTS {
        result.push(CudaFloat1 {
            0: float1 {
                x: rng.gen_range(0.0, SPACE),
            }
        });
    }
    result
}



