//! Prints the zvec runtime version. Useful for debugging which zvec C++
//! library was actually linked into the binary.

fn main() {
    println!("zvec-bindings crate version: {}", env!("CARGO_PKG_VERSION"));

    let v = zvec_bindings::version();
    println!("zvec runtime version:       {}", v);

    let (maj, min, patch) = zvec_bindings::version_tuple();
    println!("zvec version tuple:          {}.{}.{}", maj, min, patch);

    println!(
        "runtime >= 0.5.0?            {}",
        zvec_bindings::check_version(0, 5, 0)
    );
    println!(
        "runtime >= 0.10.0?           {}",
        zvec_bindings::check_version(0, 10, 0)
    );
}
