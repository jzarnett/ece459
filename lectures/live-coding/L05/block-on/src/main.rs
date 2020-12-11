use futures::executor::block_on;
use std::thread::sleep;
use std::time::Duration;

async fn q() {
  sleep(Duration::from_secs(1));
  println!("World.");
}

async fn r() {
  sleep(Duration::from_secs(1));
  println!("hello");
}

async fn hello_world() {
  r().await;
  q().await;
}

#[tokio::main]
async fn main() {
  let future = hello_world();
  block_on(future);
}

