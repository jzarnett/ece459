/*
[dependencies]
im = { version = "*" }
rand = { version = "*" }

[profile.dev]
debuginfo=false
opt-level=3
*/

#![allow(unused_imports)]
use rand::Rng;
use rand::rngs::ThreadRng;
use std::time::Instant;
use std::fmt;

use std::collections::VecDeque;

use im::Vector;

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
    // interesting experiments: vec is slow on 1..3 and vecdeque is fast
    // 4..6 is surprisingly fast for vecdeque, slow for vector
    let a = rng.gen_range(4..6);
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

const N:i32 = 100000;
fn main() {
    let mut v:Vec<i32> = vec![];
    // ^ 542ms for {0,1}, 551ms for {2,3}, 544ms for {4,5}

    // let mut v:VecDeque<i32> = VecDeque::new();
    // ^ 2ms for {0,1}, 4ms for {2,3}, 136ms for {4,5}

    // let mut v:Vector<i32> = Vector::new();
    // ^ 4ms for {0,1}, 3ms for {2,3}, 6449ms for {4,5}

    let now = Instant::now();

    let mut rng = rand::thread_rng();
    let mut v_length = 0;
    let mut max_length = 0;

    for _ in 1..100000 {
        v.insert(0, rng.gen::<i32>());
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

    let elapsed_time = now.elapsed();
    println!("Running time: {} ms", elapsed_time.as_millis());
    println!("Max length: {}", max_length);
}

