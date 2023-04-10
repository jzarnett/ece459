// try this on https://godbolt.org with `-C overflow-checks=n -C opt-level=1`

pub fn try_inline(x: i32) -> i32 {
    f(x)
}

#[inline(never)]
// ^ try change this to `#[inline(always)]`
pub fn f(x: i32) -> i32 {
    x * x
}
