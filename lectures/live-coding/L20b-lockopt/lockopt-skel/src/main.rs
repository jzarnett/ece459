/*

[dependencies]
*/

use std::sync::{Arc, Mutex};
use std::thread;

/// Count 'a', 'b', and 'c' appearances
///
/// Task: the program will create two threads. One thread will count the first
/// string and the other will count the second. However, they increment shared
/// counters. Try the program with different workloads and think how you can
/// optimize the locks given the workloads.
fn main() {
    // workloads, each workload contains two strings
    let strings = [
        [
            "hcbacbafcbaccccefdfgabcbadbabaabchccgccgabcchabccgbaebafeababgc",
            "bababaebaedeabbadgfdababdabhhabehabddfdabhbadabebahebafge",
        ],
        [
            "hdecabfcabcchebacbaddbaccdccfbaabhbacdcbaccgabab",
            "hefhddhgfabgdebabafababhfababbagababhbahffabfebaba",
        ],
        [
            "cdabchccfcbacdcccabcbaggabchbaeccehbacabeccdchefbabaabdcababgdcbaabeab",
            "hehbabafhbaabegheddabbaababhbaabhbahfdefababbababaeddge",
        ],
        [
            "cgcfchababcdgcbahcabbacdfecfabccccbaabhdgfdabfbababahbacc",
            "efeabbaeabbagggeeabegdbafbadefhfegbaabfefeab",
        ],
    ];

    let count_a = Arc::new(Mutex::new(0));
    let count_b = Arc::new(Mutex::new(0));
    let count_c = Arc::new(Mutex::new(0));

    thread::scope(|s| {
        for i in 0..2 {
            let string = &strings[0][i];
            // TODO: ^ try change 0 to 1,2,3

            let count_a_t = count_a.clone();
            let count_b_t = count_b.clone();
            let count_c_t = count_c.clone();
            s.spawn(move || {
                for c in string.chars() {
                    match c {
                        'a' => {
                            let mut count = count_a_t.lock().unwrap();
                            *count += 1;
                        }
                        'b' => {
                            let mut count = count_b_t.lock().unwrap();
                            *count += 1;
                        }
                        'c' => {
                            let mut count = count_c_t.lock().unwrap();
                            *count += 1;
                        }
                        _ => {}
                    }
                }
            });
        }
    });

    let count_a = count_a.lock().unwrap();
    let count_b = count_b.lock().unwrap();
    let count_c = count_c.lock().unwrap();

    println!("Number of 'a's: {}", *count_a);
    println!("Number of 'b's: {}", *count_b);
    println!("Number of 'c's: {}", *count_c);
}
