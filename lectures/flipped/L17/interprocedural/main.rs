fn main() {
    let mut to: &dyn Foo = &Bar;
    if foo::flag() {
        to = &Baz;
    }
    to.foo();
}

trait Foo {
    fn foo(&self) -> i32;
}

struct Bar;
impl Foo for Bar {
    fn foo(&self) -> i32 {
        println!("bar");
        0
    }
}

struct Baz;
impl Foo for Baz {
    fn foo(&self) -> i32 {
        println!("baz");
        1
    }
}
