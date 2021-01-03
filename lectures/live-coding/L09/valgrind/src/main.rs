use std::thread;
use std::time::Duration;

fn main() {

    let handle = thread::spawn(|| {
        for i in 1..10 {
            let v = Box::new(42);
            println!("hi number {} from the spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
            std::mem::forget(v)
        }
    });

    for i in 1..5 {
        println!("hi number {} from the main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }

    println!("Main thread has finished, and will wait for the spawned thread.");
    handle.join().unwrap();
    println!("Main thread has finished, and so has the thread.");
}