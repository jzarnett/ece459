/*

[dependencies]
rand = "0.7.3"
*/

use rand::seq::SliceRandom;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    for _ in 0..4 {
        let mut string = String::new();
        /// So you know 'a' and 'b' always appear together
        /// Thus only one lock for either is enough
        /// increment to 'a' counter and be applied to 'b' counter as well
        for _ in 0..20 {
            string.push('a');
            string.push('b');
        }

        /// only one string has 'c', so there will not be any race condition
        /// no lock is needed for 'c'
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
        /// So you know 'a' and 'b' always appear together
        /// Thus only one lock for either is enough
        /// increment to 'a' counter and be applied to 'b' counter as well
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
