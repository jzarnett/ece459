// try this on https://godbolt.org with `-C overflow-checks=n -C opt-level=1`

pub fn loop_unrol(a: &[i32; 4]) -> i32 {
    let mut sum = 0;
    for i in 0..4 {
        sum += &a[i];
    }
    sum
}
