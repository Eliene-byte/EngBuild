//! CADKit entry point.
//!
//! All the real work lives in the `cadkit` library crate so it can be tested
//! headlessly; this binary only calls [`cadkit::main_loop`].

fn main() {
    match cadkit::main_loop() {
        Ok(Some(adapter)) => {
            // The window is gone by the time we get here, so this is only
            // useful when the app exits cleanly.
            eprintln!("cadkit: adapter was {adapter}");
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("cadkit: {e}");
            std::process::exit(1);
        }
    }
}
