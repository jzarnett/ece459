fn main() {
    let x1 = 10;
    let y1 = 3;
    let mut r1 = 0;
    let x2 = 10f64;
    let y2 = 3f64;
    let mut r2 = 0f64;

    for _i in 0..100000000 {
        r1 += int_math(x1, y1);
        r2 += float_math(x2, y2);
    }
    println!{"({}, {})", r1, r2}
}

fn int_math(x: i64, y: i64) -> i64 {
    let mut r1 = int_power(x, y);
    r1 += int_math_helper(x, y);
    return r1;
}

fn int_math_helper(x: i64, y: i64) -> i64 {
    let r1 = x/y * int_power(y, x) / int_power(x, y);
    return r1;
}

fn int_power(x: i64, y: i64) -> i64 {
    let mut r = x;
    for _i in 1..y {
        r = r * x;
    }
    return r;
}

fn float_math(x: f64, y: f64) -> f64 {
    let mut r1 = float_power(x, y);
    r1 += float_math_helper(x, y);
    return r1;
}

fn float_math_helper(x: f64, y: f64) -> f64 {
    let r1 = x/y * float_power(y, x) / float_power(x, y);
    return r1;
}

fn float_power(x: f64, y: f64) -> f64 {
    let mut r = x;
    for _i in 1..(y as i64) {
        r = r * x;
    }
    return r;
}
