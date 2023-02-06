Make use of `for_each_with` to combine data parallelization with message passing

```rust
/*
[dependencies]
rayon = "1.6.1"
*/

use std::sync::mpsc::channel;
use rayon::prelude::*;

fn main() {
    let (tx, rx) = channel();
    println!("with rayon and message passing");
    (0..10).into_par_iter().for_each_with(tx, |tx, x| {
        tx.send(x).unwrap();
        println!("{:?}", x);
    });

    let received: Vec<i32> = rx.iter().collect();
    println!("{:?}", received);
}
```
