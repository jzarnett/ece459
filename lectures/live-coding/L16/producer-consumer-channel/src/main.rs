use rand::Rng;
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use crossbeam_channel::unbounded;

const CHANNEL_CAPACITY: usize = 100;
const NUM_THREADS: usize = 4;
const ITEMS_PER_THREAD: usize = 10000;

fn main() {
    let (send_end, receive_end) = unbounded();

    let mut threads = vec![];
    for _i in 0 .. NUM_THREADS {
        let send_end = send_end.clone();
        threads.push(
            thread::spawn(move || {
                for _k in 0 .. ITEMS_PER_THREAD {
                    let produced_value = produce_item();
                    send_end.send(produced_value).unwrap();
                }
            })
        );
    }

    for _j in 0 .. NUM_THREADS {
        // create consumers
        let receive_end = receive_end.clone();
        threads.push(
            thread::spawn(move || {
                for _k in 0 .. ITEMS_PER_THREAD {
                    let to_consume = receive_end.recv().unwrap();
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