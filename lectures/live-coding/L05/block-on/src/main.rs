use futures::executor::block_on;

async fn q() -> u32 {
  return 42;
}

async fn hello_world() {
  q().await;
  println!("hello");
}

fn main() {
  let future = hello_world();
  block_on(future);
}

