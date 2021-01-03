use std::thread;
use std::sync::{Mutex, Arc};

fn main() {
    let a = Arc::new(Mutex::new(0));
    let b = Arc::new(Mutex::new(0));
    let a2 = a.clone();
    let b2 = b.clone();

    let handle = thread::spawn(move|| {
        *a.lock().unwrap() = 1;
        *b.lock().unwrap() = 1;
    });
    let handle2 = thread::spawn(move|| {
        *b2.lock().unwrap() = 1;
        *a2.lock().unwrap() = 1;
    });

    handle.join().unwrap();
    handle2.join().unwrap();
}