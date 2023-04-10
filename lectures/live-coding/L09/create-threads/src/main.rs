use std::thread;

fn main() {
    for _ in 0 .. 2000 {
        thread::spawn(|| {
            false
        });
    }
}
