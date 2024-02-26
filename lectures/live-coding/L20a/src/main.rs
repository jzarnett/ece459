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
    let a = 1;
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

const N:i32 = 10_000;
fn main() {
    let mut v:Vec<i32> = vec![];
    // TODO: add times

    // let mut v:VecDeque<i32> = VecDeque::new();
    // TODO: add times

    // let mut v:Vector<i32> = Vector::new();
    // TODO: add times

    let mut rng = rand::thread_rng();
    let mut v_length = 0;
    let mut max_length = 0;

    for _ in 0..100_000 {
        v.insert(v_length, rng.gen::<i32>());
        v_length = v_length + 1;
    }

    let now = Instant::now();

    for _n in 0..N {
        let action = generate_action(&mut rng);
        match action {
            PushEnd(i) => { v.insert(v_length, i); v_length = v_length + 1; },
            PopEnd => { if ! v.is_empty() { v_length = v_length - 1; v.remove(v_length); } },
            PushStart(i) => { v.insert(0, i); v_length = v_length + 1 },
            PopStart => { if ! v.is_empty() { v_length = v_length - 1; v.remove(0); } },
            PushRandom(i) => { v.insert(rng.gen_range(0..v_length), i); v_length = v_length + 1 },
            PopRandom => { if ! v.is_empty() { v.remove(rng.gen_range(0..v_length)); v_length = v_length - 1 } },
        }
        if v_length > max_length { max_length = v_length; }
    }

    let elapsed_time = now.elapsed();
    println!("Running time: {} micro seconds", elapsed_time.as_micros());
    println!("Max length: {}", max_length);
}

