#[macro_use]
extern crate rustacuda;

use cuda_sys::vector_types::{float3, float4};
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use rand::Rng;

/* A Rustification by Jeff Zarnett of a past ECE 459 N-Body assignment that was
originally from GPU Gems, Chapter 31 and modified by Patrick Lam.
Then CUDA-fied by Jeff Zarnett using the Rustacuda library example code
 */

const NUM_POINTS: u32 = 100000;
const SPACE: f32 = 1000.0;

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let initial_positions = initialize_positions();
    println! {"Initial positions:"}
    for pt in initial_positions.iter() {
        println! {"({}, {}, {}) [{}]", pt.x, pt.y, pt.z, pt.w};
    }

    let ptx = CString::new(include_str!("../resources/nbody.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10])?;
    let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
    let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
    let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10])?;

    // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.calculate_forces<<<1, 1, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_1.as_device_ptr(),
            out_1.len()
        ));
        result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 20];
    out_1.copy_to(&mut out_host[0..10])?;
    out_2.copy_to(&mut out_host[10..20])?;

    for x in out_host.iter() {
        assert_eq!(3.0 as u32, *x as u32);
    }

    // let final_accelerations = calculate_forces(initial_positions);
    // println! {"Accelerations:"}
    // for accel in final_accelerations.iter() {
    //     println! {"{}", accel};
    // }
    Ok(())
}

fn initialize_positions() -> Vec<float4> {
    let mut result: Vec<float4> = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..NUM_POINTS {
        result.push(float4 {
            x: rng.gen_range(0.0, SPACE),
            y: rng.gen_range(0.0, SPACE),
            z: rng.gen_range(0.0, SPACE),
            w: rng.gen_range(0.01, 100.0),
        });
    }
    result
}

fn initialize_accelerations() -> Vec<float3> {
    let mut result: Vec<float3> = Vec::new();
    for _i in 0 .. NUM_POINTS {
        result.push(float3 {
            x: 0f32,
            y: 0f32,
            z: 0f32,
        })
    }
    result
}


