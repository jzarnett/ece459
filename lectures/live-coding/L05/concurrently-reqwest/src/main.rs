async fn get_webpage_1() -> Result<(), Box<dyn std::error::Error>> {
    let resp = reqwest::get("https://www.rust-lang.org")
        .await?
        .text()
        .await?;
    println!("{:#?}", resp);
    Ok(())
}

async fn get_webpage_2() -> Result<(), Box<dyn std::error::Error>> {
    let resp = reqwest::get("https://www.uwaterloo.ca")
        .await?
        .text()
        .await?;
    println!("{:#?}", resp);
    Ok(())
}

#[tokio::main]
async fn main() {
    let f1 = get_webpage_1();
    let f2 = get_webpage_2();

    futures::join!(f1, f2);
}
