use std::{sync::{Mutex, MutexGuard}, thread};
use std::thread::sleep;
use std::time::Duration;

use lazy_static::lazy_static;
lazy_static! {
    static ref MUTEX1: Mutex<i64> = Mutex::new(0);
    static ref MUTEX2: Mutex<i64> = Mutex::new(0);
}

fn main() {
    // Spawn thread and store handles
    let mut children = vec![];
    for i_thread in 0..2 {
        children.push(thread::spawn(move || {
            for _ in 0..1 {
                // Thread 1
                if i_thread % 2 == 0 {
                    // Lock mutex1
                    // No need to specify type but yes create a dummy variable to prevent rust
                    // compiler from being lazy
                    let _guard: MutexGuard<i64> = MUTEX1.lock().unwrap();

                    // Just log
                    println!("Thread {} locked mutex1 and will try to lock the mutex2, after a nap !", i_thread);

                    // Here I sleep to let Thread 2 lock mutex2
                    sleep(Duration::from_millis(10));

                    // Lock mutex 2
                    let _guard = MUTEX2.lock().unwrap();
                // Thread 2
                } else {
                    // Lock mutex 1
                    let _guard = MUTEX2.lock().unwrap();

                    println!("Thread {} locked mutex2 and will try to lock the mutex1", i_thread);

                    // Here I freeze !
                    let _guard = MUTEX1.lock().unwrap();
                }
            }
        }));
    }

    // Wait
    for child in children {
        let _ = child.join();
    }

    println!("Want to see this after converting to trylocks");
}
