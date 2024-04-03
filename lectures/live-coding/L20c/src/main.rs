#![allow(unused_imports)]
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
    let a = 0;
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
const INITIAL_CAPACITY:usize = 1;
const FIXED_INITIAL_VEC_SIZE:i32 = 5000000;
const N:i32 = 100;
fn main() {
    let mut max_length = 0;
    let now = Instant::now();
    for _ in 0..OUTER {
        let mut v:Vec<i32> = Vec::with_capacity(INITIAL_CAPACITY);

        let mut rng = rand::thread_rng();
        let mut v_length = 0;

        for _ in 0..FIXED_INITIAL_VEC_SIZE {
            v.insert(v_length, rng.gen::<i32>());
            v_length = v_length + 1;
        }

        for _n in 0..N {
            let action = generate_action(&mut rng);
            match action {
                PushEnd(i) => { v.insert(v_length, i); v_length = v_length + 1; },
                PopEnd => { if ! v.is_empty() { v_length = v_length - 1; v.remove(v_length); } },
                PushStart(i) => { v.insert(0, i); v_length = v_length + 1 },
                PopStart => { if ! v.is_empty() { v_length = v_length - 1; v.remove(0); } },
                PushRandom(i) => { v.insert(rng.gen_range(0..=v_length), i); v_length = v_length + 1 },
                PopRandom => { if ! v.is_empty() { v.remove(rng.gen_range(0..v_length)); v_length = v_length - 1 } },
            }
            if v_length > max_length { max_length = v_length; }
        }

    }
    let elapsed_time = now.elapsed();
    println!("Running time: {} ms", elapsed_time.as_millis());
    println!("Max length: {}", max_length);

    // modify main.rs line "const INITIAL_CAPACITY:usize = 1";

    let target_line_start = "const INITIAL_CAPACITY:usize =";
    let new_target_line = format!("{} {};", target_line_start, max_length);

    let mut new_lines: String = "".to_owned();
    if let Ok(lines) = read_lines("src/main.rs") {
        // Consumes the iterator, returns an (Optional) String
        for line in lines {
            if let Ok(ip) = line {
                if ip.starts_with(target_line_start) {
                    new_lines.push_str(&new_target_line);
                } else {
                    new_lines.push_str(&ip);
                }
                new_lines.push_str("\n");
            }
        }
    }
    fs::write("src/main.rs", new_lines).expect("can't write output");
    // recompile
    println!("-- recompile");
    Command::new("cargo")
            .arg("build")
            .arg("--release")
            .output()
            .expect("failed to execute process");
    println!("-- done. rerun to inspect the time change");
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

