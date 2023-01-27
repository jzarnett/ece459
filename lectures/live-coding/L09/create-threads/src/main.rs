use std::thread;

fn main() {
    for _ in 0 .. 50000 {
        thread::spawn(|| {
            false
        });
    }
}
