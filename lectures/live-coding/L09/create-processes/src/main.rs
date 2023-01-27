use std::process::Command;

fn main() {
    for j in 0 .. 50000 {
       Command::new("/bin/false").spawn();
    }
}
