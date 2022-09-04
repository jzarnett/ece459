use rand::Rng;
use tokio::sync::Semaphore;
use std::sync::{Mutex, Arc};
use std::thread;
use futures::executor::block_on;
use std::thread::sleep;
use std::time::Duration;

const BUFFER_SIZE: usize = 100;
const ITEMS_PER_THREAD: usize = 10000;
const NUM_THREADS: usize = 4;

struct SharedBuffer {
    buffer: Vec<i32>,
    producer_count: usize,
    consumer_count: usize
}

fn main() {
    let spaces = Arc::new(Semaphore::new(BUFFER_SIZE));
    let items = Arc::new(Semaphore::new(0));
    let shared_buffer = SharedBuffer {
        buffer: vec![-1; BUFFER_SIZE],
        producer_count: 0 as usize,
        consumer_count: 0 as usize
    };
    let buffer = Arc::new(Mutex::new(shared_buffer));

    let mut threads = vec![];
    for _i in 0 .. NUM_THREADS {
        // create producers
        let spaces = spaces.clone();
        let items = items.clone();
        let buffer = buffer.clone();
        threads.push(
            thread::spawn(move || {
                for _k in 0 .. ITEMS_PER_THREAD {
                    let produced_value = produce_item();
                    let permit = block_on(spaces.acquire());
                    {
                        let mut buf = buffer.lock().unwrap();
                        let current_produce_space = buf.producer_count;
                        let next_produce_space = (current_produce_space + 1) % buf.buffer.len();
                        buf.buffer.insert(current_produce_space, produced_value);
                        buf.producer_count = next_produce_space;
                    }
                    items.add_permits(1);
                    permit.expect("Permit Exists").forget();
                }
            })
        );
    }

    for _j in 0 .. NUM_THREADS {
        // create consumers
        let spaces = spaces.clone();
        let items = items.clone();
        let buffer = buffer.clone();
        threads.push(
            thread::spawn(move || {
                for _k in 0 .. ITEMS_PER_THREAD {
                    let permit = block_on(items.acquire());
                    let to_consume = {
                        let mut buf = buffer.lock().unwrap();
                        let current_consume_space = buf.consumer_count;
                        let next_consume_space = (current_consume_space + 1) % buf.buffer.len();
                        let to_consume = *buf.buffer.get(current_consume_space).unwrap();
                        buf.consumer_count = next_consume_space;
                        to_consume
                    };
                    spaces.add_permits(1);
                    permit.expect("Permit Exists").forget();
                    consume_item(to_consume);
                }
            })
        );
    }

    for t in threads {
        let _ = t.join();
    }
    println!{"Done!"}
}

fn produce_item() -> i32 {
    let item = rand::thread_rng().gen_range(0, 100000);
    sleep(Duration::new(0, 5000));
    println!("Produced item {}", item);
    item
}

fn consume_item(item: i32) {
    sleep(Duration::new(0, 5000));
    println!("Consumed item {}", item);
}