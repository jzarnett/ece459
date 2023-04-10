use std::process::Command;

fn main() {
    for _ in 0 .. 2000 { // The number cannot be too large
       Command::new("/usr/bin/false").spawn().unwrap();
    }
}
