use std::thread;
use std::sync::{Mutex, Arc};
use std::time::Duration;

fn main() {
    let a = Arc::new(Mutex::new(0));
    let b = Arc::new(Mutex::new(0));
    let a2 = a.clone();
    let b2 = b.clone();
    let a3 = a.clone();
    let b3 = b.clone();

    let handle = thread::spawn(move|| {
        let mut a = a3.lock().unwrap();
        let mut b = b3.lock().unwrap();
        *a = 2;
        *b = 2;
    });
    let handle2 = thread::spawn(move|| {
        let mut b = b2.lock().unwrap();
        let mut a = a2.lock().unwrap();
        *b = 1;
        *a = 1;
    });

    handle.join().unwrap();
    handle2.join().unwrap();
    println!{"Final Values: {}, {}", *a.lock().unwrap(), *b.lock().unwrap() }
}