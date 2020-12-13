fn g() {
    let a = Vec::<u32>::with_capacity(4000);
    std::mem::forget(a)
}

fn f() {
    let a = Vec::<u32>::with_capacity(2000);
    std::mem::forget(a);
    g()
}

fn main() {

    let mut a = Vec::with_capacity(10);
    for _i in 0..10 {
	a.push(Box::new([0;1000]))
    }
    f();
    g();
}
