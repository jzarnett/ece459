use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    let handle = thread::spawn(move || {
        println!("Here’s a vector: {:?}", v);
        v
    });
    let v = handle.join().expect("Problem returning a value");
    println!("Here’s a vector: {:?}", v);
}