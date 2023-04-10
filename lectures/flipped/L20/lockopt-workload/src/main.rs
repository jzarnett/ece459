/*

[dependencies]
rand = "0.7.3"
*/

use rand::seq::SliceRandom;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    for _ in 0..6 {
        let mut string = String::new();
        for _ in 0..20 {
            string.push('a');
            string.push('b');
        }

        for _ in 0..10 {
            string.push('c');
        }
        let extra_chars = vec!['d', 'e', 'f', 'g', 'h'];
        for _ in 0..20 {
            string.push(extra_chars[rng.gen_range(0, extra_chars.len())]);
        }
        unsafe {
            string.as_mut_vec().shuffle(&mut rng);
        }
        println!("[\n{:?}", string);

        let mut string = String::new();
        for _ in 0..20 {
            string.push('a');
            string.push('b');
        }
        let extra_chars = vec!['d', 'e', 'f', 'g', 'h'];
        for _ in 0..30 {
            string.push(extra_chars[rng.gen_range(0, extra_chars.len())]);
        }
        unsafe {
            string.as_mut_vec().shuffle(&mut rng);
        }
        println!("{:?}\n],", string);
    }
}
