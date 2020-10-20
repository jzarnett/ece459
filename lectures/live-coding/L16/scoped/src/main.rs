fn main() {
    let v = vec![1, 2, 3];
    println!("main thread has id {}", thread_id::get());

    crossbeam::scope(|scope| {
        scope.spawn(|inner_scope| {
            println!("Here's a vector: {:?}", v);
            println!("Now in thread with id {}", thread_id::get());
        });
    }).unwrap();

    println!("Vector v is back: {:?}", v);
}