use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_ZVEC_GIT_REF: &str = "v0.5.0";

fn zvec_git_ref() -> String {
    env::var("ZVEC_GIT_REF").unwrap_or_else(|_| DEFAULT_ZVEC_GIT_REF.to_string())
}

fn ensure_zvec_source(workspace_dir: &Path) -> PathBuf {
    let zvec_src = workspace_dir.join("vendor/zvec");
    let git_ref = zvec_git_ref();

    if zvec_src.join("CMakeLists.txt").exists() {
        println!("cargo:warning=zvec source already present");
        return zvec_src;
    }

    println!(
        "cargo:warning=Cloning zvec {} (this may take a few minutes)...",
        git_ref
    );
    let _ = std::fs::create_dir_all(zvec_src.parent().unwrap());

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--branch",
            &git_ref,
            "--recursive",
            "https://github.com/alibaba/zvec.git",
            zvec_src.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute git clone. Please ensure git is installed.");

    if !status.success() {
        panic!("git clone failed. Please check your network connection and that git is installed.");
    }

    zvec_src
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir.parent().expect("Expected parent directory");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    println!("cargo:rerun-if-env-changed=ZVEC_GIT_REF");
    println!("cargo:rerun-if-env-changed=ZVEC_BUILD_TYPE");
    println!("cargo:rerun-if-env-changed=ZVEC_BUILD_PARALLEL");
    println!("cargo:rerun-if-env-changed=ZVEC_CPU_ARCH");
    println!("cargo:rerun-if-env-changed=ZVEC_OPENMP");

    let zvec_src = ensure_zvec_source(workspace_dir);
    let zvec_build = zvec_src.join("build");
    let zvec_lib = zvec_build.join("lib");

    let build_type = env::var("ZVEC_BUILD_TYPE").unwrap_or_else(|_| "Release".to_string());
    let parallel_jobs = env::var("ZVEC_BUILD_PARALLEL")
        .map(|s| s.parse::<usize>().unwrap_or_else(|_| num_cpus()))
        .unwrap_or_else(|_| num_cpus());

    // Overlay directory that compiles upstream c_api.cc as a static library.
    let c_api_overlay_dir = manifest_dir.join("c-api-static");
    let c_api_build = out_dir.join("c-api-static-build");

    // Group-by shim directory (wraps zvec::Collection::GroupByQuery since the
    // upstream C API does not expose group-by in v0.5.0).
    let groupby_shim_dir = manifest_dir.join("groupby-shim");
    let groupby_shim_build = out_dir.join("groupby-shim-build");

    // zvec C++ libraries (produces libzvec.a, libzvec_core.a, etc.)
    // In v0.5.0 the all-in-one library is `libzvec.a` (was `libzvec_db.a` in v0.2.1).
    let zvec_built = zvec_lib.join("libzvec.a");
    if !zvec_built.exists() {
        println!("cargo:warning=Building zvec C++ library...");
        build_zvec(&zvec_src, &zvec_build, &build_type, parallel_jobs);
    } else {
        println!("cargo:warning=zvec C++ library already built");
    }

    // Build the upstream c_api.cc as a static library (libzvec_c_api_static.a).
    let c_api_built = c_api_build.join("libzvec_c_api_static.a");
    if !c_api_built.exists() {
        println!("cargo:warning=Building zvec_c_api_static (compiling upstream c_api.cc)...");
        build_c_api_static(
            &c_api_overlay_dir,
            &c_api_build,
            &zvec_src,
            &zvec_build,
            &build_type,
            parallel_jobs,
        );
    } else {
        println!("cargo:warning=zvec_c_api_static already built");
    }

    // Build the group-by shim (libzvec_groupby_shim.a).
    let groupby_built = groupby_shim_build.join("libzvec_groupby_shim.a");
    if !groupby_built.exists() {
        println!("cargo:warning=Building zvec_groupby_shim...");
        build_groupby_shim(
            &groupby_shim_dir,
            &groupby_shim_build,
            &zvec_src,
            &build_type,
            parallel_jobs,
        );
    } else {
        println!("cargo:warning=zvec_groupby_shim already built");
    }

    generate_bindings(&zvec_src, &zvec_build, &groupby_shim_dir);

    link_libraries(&zvec_lib, &c_api_build, &groupby_shim_build);
}

fn build_zvec(_src: &Path, build: &Path, build_type: &str, parallel_jobs: usize) {
    let _ = std::fs::create_dir_all(build);

    let mut cmake_args = vec![
        format!("-DCMAKE_BUILD_TYPE={}", build_type),
        "-DBUILD_PYTHON_BINDINGS=OFF".to_string(),
        "-DBUILD_TOOLS=OFF".to_string(),
        // C bindings are built as a shared library by upstream; we don't use that
        // target (we compile c_api.cc ourselves via the c-api-static overlay), but
        // we still enable it so the configured c_api.h header is generated.
        "-DBUILD_C_BINDINGS=ON".to_string(),
    ];

    if let Ok(arch) = env::var("ZVEC_CPU_ARCH") {
        cmake_args.push(format!("-DENABLE_{}=ON", arch));
    }

    if env::var("ZVEC_OPENMP")
        .map(|v| v == "ON" || v == "1")
        .unwrap_or(false)
    {
        cmake_args.push("-DENABLE_OPENMP=ON".to_string());
    }

    cmake_args.push(format!("-S{}", _src.display()));
    cmake_args.push(format!("-B{}", build.display()));

    run(
        Command::new("cmake").args(&cmake_args),
        "cmake configure for zvec",
    );

    run(
        Command::new("cmake")
            .args([
                "--build",
                build.to_str().expect("Invalid build path"),
                "-j",
                parallel_jobs.to_string().as_str(),
            ]),
        "build zvec",
    );
}

/// Compile upstream zvec `src/binding/c/c_api.cc` as a STATIC library using
/// our overlay CMakeLists. This avoids upstream's SHARED-only default while
/// preserving the project's static-linking deployment story.
fn build_c_api_static(
    overlay_dir: &Path,
    build: &Path,
    zvec_src: &Path,
    zvec_build: &Path,
    build_type: &str,
    parallel_jobs: usize,
) {
    let _ = std::fs::create_dir_all(build);

    run(
        Command::new("cmake").args([
            format!("-S{}", overlay_dir.display()).as_str(),
            format!("-B{}", build.display()).as_str(),
            format!("-DZVEC_SRC={}", zvec_src.display()).as_str(),
            format!("-DZVEC_BUILD={}", zvec_build.display()).as_str(),
            format!("-DCMAKE_BUILD_TYPE={}", build_type).as_str(),
        ]),
        "cmake configure for zvec_c_api_static",
    );

    run(
        Command::new("cmake")
            .args([
                "--build",
                build.to_str().expect("Invalid c-api-static build path"),
                "-j",
                parallel_jobs.to_string().as_str(),
            ]),
        "build zvec_c_api_static",
    );
}

/// Build the group-by shim (wraps zvec::Collection::GroupByQuery, which is
/// not exposed by the upstream C API in v0.5.0).
fn build_groupby_shim(
    shim_dir: &Path,
    build: &Path,
    zvec_src: &Path,
    build_type: &str,
    parallel_jobs: usize,
) {
    let _ = std::fs::create_dir_all(build);

    run(
        Command::new("cmake").args([
            format!("-S{}", shim_dir.display()).as_str(),
            format!("-B{}", build.display()).as_str(),
            format!("-DZVEC_SRC={}", zvec_src.display()).as_str(),
            format!("-DCMAKE_BUILD_TYPE={}", build_type).as_str(),
        ]),
        "cmake configure for zvec_groupby_shim",
    );

    run(
        Command::new("cmake")
            .args([
                "--build",
                build.to_str().expect("Invalid groupby-shim build path"),
                "-j",
                parallel_jobs.to_string().as_str(),
            ]),
        "build zvec_groupby_shim",
    );
}

fn generate_bindings(zvec_src: &Path, zvec_build: &Path, groupby_shim_dir: &Path) {
    // Use the CONFIGURED c_api.h produced by upstream's cmake configure_file.
    // The source header has @ZVEC_VERSION_MAJOR@ placeholders that need substitution.
    let c_api_header = zvec_build.join("src/generated/zvec/c_api.h");
    let groupby_header = groupby_shim_dir.join("include/zvec_groupby_shim.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    if !c_api_header.exists() {
        panic!(
            "Configured c_api.h not found at {}. Run cmake configure for zvec first.",
            c_api_header.display()
        );
    }
    if !groupby_header.exists() {
        panic!(
            "Group-by shim header not found at {}.",
            groupby_header.display()
        );
    }

    // Write a combined wrapper header so bindgen processes both APIs in one pass.
    let wrapper_header = out_path.join("wrapper.h");
    std::fs::write(
        &wrapper_header,
        format!(
            "#include \"{}\"\n#include \"{}\"\n",
            c_api_header.display(),
            groupby_header.display()
        ),
    )
    .expect("Failed to write wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(wrapper_header.to_str().expect("Invalid wrapper header path"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate_comments(true)
        .allowlist_function("zvec_.*")
        .allowlist_function("zvecgb_.*")
        .allowlist_type("zvec_.*")
        .allowlist_type("zvecgb_.*")
        .allowlist_var("ZVEC_.*")
        .blocklist_type("__.*")
        .default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
        .clang_arg(format!("-I{}", zvec_build.join("src/generated").display()))
        .clang_arg(format!("-I{}", zvec_src.join("src/include").display()))
        .clang_arg(format!(
            "-I{}",
            groupby_shim_dir.join("include").display()
        ));

    // Best-effort system include paths (only add if they exist on this machine).
    for path in ["/usr/include", "/usr/local/include"] {
        if Path::new(path).exists() {
            builder = builder.clang_arg(format!("-I{}", path));
        }
    }

    // Discover GCC's compiler-specific include dir (hosts stdbool.h, stddef.h, etc.)
    // by parsing `gcc -E -Wp,-v` output. Avoids hard-coding aarch64-linux-gnu/13.
    if let Ok(gcc_includes) = discover_gcc_include_dirs() {
        for path in gcc_includes {
            if Path::new(&path).exists() {
                builder = builder.clang_arg(format!("-I{}", path));
            }
        }
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_libraries(zvec_lib: &Path, c_api_build: &Path, groupby_shim_build: &Path) {
    // Static archive of upstream c_api.cc (compiled by our overlay)
    println!(
        "cargo:rustc-link-search=native={}",
        c_api_build.display()
    );
    println!("cargo:rustc-link-lib=static:+whole-archive=zvec_c_api_static");

    // Group-by shim
    println!(
        "cargo:rustc-link-search=native={}",
        groupby_shim_build.display()
    );
    println!("cargo:rustc-link-lib=static:+whole-archive=zvec_groupby_shim");

    // zvec component libraries path
    println!("cargo:rustc-link-search=native={}", zvec_lib.display());

    // External third-party libraries (built in build/external/usr/local/lib)
    let external_lib = zvec_lib.parent().unwrap().join("external/usr/local/lib");
    println!("cargo:rustc-link-search=native={}", external_lib.display());

    // Arrow build directory (contains thrift and many other libs)
    let arrow_build = zvec_lib
        .parent()
        .unwrap()
        .join("thirdparty/arrow/arrow/src/ARROW.BUILD-build");
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("release").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("re2_ep-install/lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("utf8proc_ep-install/lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build
            .join("zlib_ep/src/zlib_ep-install/lib")
            .display()
    );

    // Boost libraries
    let boost_build = arrow_build.join("_deps/boost-build/libs");
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("atomic").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("charconv").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("chrono").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("container").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("date_time").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("locale").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("thread").display()
    );

    // LZ4
    let lz4_build = zvec_lib
        .parent()
        .unwrap()
        .join("thirdparty/lz4/lz4/src/Lz4.BUILD/lib");
    println!("cargo:rustc-link-search=native={}", lz4_build.display());

    // zvec C++ libraries as whole-archive to ensure they're linked in tests
    // (Cargo doesn't propagate regular static lib links to test binaries).
    // In v0.5.0 the all-in-one is `zvec` (was `zvec_db` in v0.2.1).
    // Note: libzvec.a bundles zvec_common, zvec_index, zvec_proto, zvec_sqlengine.
    // Note: libzvec_core.a bundles core_knn_* libraries.
    let whole_archive_libs = ["zvec", "zvec_core", "zvec_ailego", "zvec_turbo"];
    for lib in &whole_archive_libs {
        println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
    }

    // Third-party dependencies (whole-archive for test linking).
    // Note: 'z', 'utf8proc', 're2', 'thrift' are included in arrow_bundled_dependencies.
    // FastPFOR is needed by v0.5.0 FTS (bit-packing for posting lists).
    let thirdparty_libs = [
        "parquet",
        "arrow_acero",
        "arrow_dataset",
        "arrow_compute",
        "arrow",
        "arrow_bundled_dependencies",
        "roaring",
        "rocksdb",
        "lz4",
        "protobuf",
        "protoc",
        "boost_thread",
        "boost_atomic",
        "boost_chrono",
        "boost_container",
        "boost_date_time",
        "boost_locale",
        "boost_charconv",
        "glog",
        "gflags_nothreads",
        "antlr4-runtime",
        "FastPFOR",
    ];
    for lib in &thirdparty_libs {
        println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
    }

    // System libraries
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
}

fn run(cmd: &mut Command, context: &str) {
    println!("cargo:warning=Running: {:?}", cmd);
    let status = cmd.status().unwrap_or_else(|_| {
        panic!("Failed to execute command: {}", context);
    });
    if !status.success() {
        panic!("Command failed ({}): {:?}", context, cmd);
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Discover GCC's system include directories by parsing `gcc -E -Wp,-v` output.
/// This is where stdbool.h, stddef.h, etc. live. Path varies by platform
/// (e.g. `/usr/lib/gcc/aarch64-linux-gnu/13/include` on arm64 Debian,
///  `/usr/lib/gcc/x86_64-linux-gnu/13/include` on amd64).
fn discover_gcc_include_dirs() -> std::io::Result<Vec<String>> {
    let output = Command::new("gcc")
        .args(["-E", "-Wp,-v", "-"])
        .stdin(std::process::Stdio::null())
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut dirs = Vec::new();
    let mut in_search_list = false;
    for line in stderr.lines() {
        if line.contains("#include <...> search starts here:") {
            in_search_list = true;
            continue;
        }
        if line.contains("End of search list.") {
            break;
        }
        if in_search_list {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                dirs.push(trimmed.to_string());
            }
        }
    }
    Ok(dirs)
}
