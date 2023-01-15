# In-class exercises

## q1
### reference counter
Use multiple threads to modify a String

### starter code if needed
Fix the following code so that we can modify the `s` from both threads

```rust
use std::time;
use std::thread;
use std::sync::Mutex;

fn main() {

    let s = String::from("start\n");

    let mutex = Mutex::new(s);

    let h = thread::spawn(|| {
        for _i in 0..2 {
            mutex.lock().unwrap().push_str("child thread\n");
            thread::sleep(time::Duration::from_millis(1));
        }
    });

    for _i in 0..2 {
        mutex.lock().unwrap().push_str("main thread\n");
        thread::sleep(time::Duration::from_millis(1));
    }

    h.join().expect("fail to join handle");
    println!("{}", mutex.lock().unwrap());
}
```

## q2
### lifetimes
Read
<https://doc.rust-lang.org/stable/book/ch10-03-lifetime-syntax.html#lifetime-elision>
and try to expand the following functions

```rust
fn print(s: &str);                                   // elided

fn debug(lvl: usize, s: &str);                       // elided

fn substr(s: &str, until: usize) -> &str;            // elided

fn new(buf: &mut [u8]) -> Thing;                     // elided
```

Answer is proivded on
<https://doc.rust-lang.org/reference/lifetime-elision.html?highlight=lifetime#lifetime-elision-in-functions>

## q3
### Unsafe
Try to write tiny piece of code using a raw pointer. (Exercise is based on
https://doc.rust-lang.org/stable/book/ch19-01-unsafe-rust.html#creating-a-safe-abstraction-over-unsafe-code
and Listing 19-7: Creating a slice from an arbitrary memory location)


### starter code if needed

```rust
use std::slice;

fn main() {
    let size = 5;

    // create a i32 vector with `size` capacity
    // let v = ...

    // get its `address` by `as_mut_ptr`
    // convert `address` to a raw pointer `r`

    let len = 5;
    // TODO: ^ try increase the `len` to create a segment fault
    // can `len = 6` cause a segment fault?

    let values: &[i32] = unsafe {
        slice::from_raw_parts_mut(r, len)
    };

    println!("{}", values[len - 1]);
}
```
