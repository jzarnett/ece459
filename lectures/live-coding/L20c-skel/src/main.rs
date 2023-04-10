use std::collections::VecDeque;
use rand::Rng;
use rand::rngs::ThreadRng;
use std::time::Instant;
use std::fmt;
use std::process::Command;
use std::fs;
use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead};
use std::fmt::Result;

use crate::Action::PushEnd;
use crate::Action::PopEnd;
use crate::Action::PushStart;
use crate::Action::PopStart;
use crate::Action::PushRandom;
use crate::Action::PopRandom;

enum Action {
    PushStart(i32),
    PushEnd(i32),
    PushRandom(i32),
    PopStart,
    PopEnd,
    PopRandom,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return match self {
            PushStart(_) => write!(f, "PushStart"),
            PushEnd(_) => write!(f, "PushEnd"),
            PushRandom(_) => write!(f, "PushRandom"),
            PopStart => write!(f, "PopStart"),
            PopEnd => write!(f, "PopEnd"),
            PopRandom => write!(f, "PopRandom"),
        }
    }
}

fn generate_action(rng:&mut ThreadRng) -> Action {
    let a = rng.gen_range(0..2);
    match a {
        0 => PushEnd(rng.gen::<i32>()),
        1 => PopStart,
        2 => PushStart(rng.gen::<i32>()),
        3 => PopEnd,
        4 => PushRandom(rng.gen::<i32>()),
        5 => PopRandom,
        _ => PopEnd,
    }
}

const OUTER:i32 = 200;
const INITIAL_CAPACITY:usize = 5000000;
const FIXED_INITIAL_VEC_SIZE:i32 = 5000000;
const N:i32 = 100;
fn main() {
    let mut max_length = 0;
    let now = Instant::now();
    for _ in 1..OUTER {
        let mut v:VecDeque<i32> = VecDeque::with_capacity(INITIAL_CAPACITY);

        let mut rng = rand::thread_rng();
        let mut v_length = 0;

        for _ in 1..FIXED_INITIAL_VEC_SIZE {
            v.insert(0, 2);
            v_length = v_length + 1;
        }

        for _n in 1..N {
            let action = generate_action(&mut rng);
            match action {
                PushEnd(i) => { v.insert(v_length, i); v_length = v_length + 1; },
                PopEnd => { if ! v.is_empty() { v_length = v_length - 1; v.remove(v_length); } },
                PushStart(i) => { v.insert(0, i); v_length = v_length + 1 },
                PopStart => { if ! v.is_empty() { v_length = v_length - 1; v.remove(0); } },
                PushRandom(i) => { v.insert(rng.gen_range(0..v_length+1), i); v_length = v_length + 1 },
                PopRandom => { if ! v.is_empty() { v.remove(rng.gen_range(0..v_length)); v_length = v_length - 1 } },
                _ => {}
            }
            if v_length > max_length { max_length = v_length; }
        }

    }
    let elapsed_time = now.elapsed();
    println!("Running time: {} ms", elapsed_time.as_millis());
    println!("Max length: {}", max_length);

    // modify main.rs line "const INITIAL_CAPACITY:usize = 1";

    // ...

    // recompile
    // Command::new(...);
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
