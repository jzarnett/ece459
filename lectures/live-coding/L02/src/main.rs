  fn main() {
    let mut s1 = String::from("hello");
      let len = calc_len(&mut s1);
      println!("The length of '{}' is '{}'.", s1, len);
  }

  fn calc_len(s: &mut String) -> usize {
      s.len()
  }
