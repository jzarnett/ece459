use rayon::prelude::*;
use rand::Rng;
use std::i64::MIN;
use std::i64::MAX;
use std::sync::atomic::{AtomicI64, Ordering};

const VEC_SIZE: usize = 10000000;

fn main() {
    let vec = init_vector();
    let max = AtomicI64::new(MIN);
    vec.par_iter().for_each(|n| {
	// this code is wrong! can you see why?
        let mut previous_value = max.load(Ordering::SeqCst);
        if *n > previous_value {
	    // wrong!
            while max.compare_and_swap(previous_value, *n, Ordering::SeqCst) != previous_value {
                println!("Compare and swap was unsuccessful; retrying");
                previous_value = max.load(Ordering::SeqCst);
            }
        }
	// L16 video describes it, lecture notes PDF has correct code
    });
    let final_max = max.load(Ordering::SeqCst);
    println!("Max value in the array is {}", final_max);
    if final_max == MAX {
        println!("This is the max value for an i64.")
    }
}

fn init_vector() -> Vec<i64> {
    let mut rng = rand::thread_rng();
    let mut vec = Vec::new();
    for _i in 0 ..VEC_SIZE {
        vec.push(rng.gen::<i64>())
    }
    vec
}
