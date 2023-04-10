# In-class exercise: using futures

/*
[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.11" }
*/

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resp = reqwest::get("https://www.rust-lang.org")
        .await?
        .text()
        .await?;
    println!("{:#?}", resp);
    Ok(())
}
