1. Rename `lib.rs` (IMPORTANT):
```
mv src/lib.rs src/lib_unexpanded.rs
```
2. Switch to nightly. This is required for some of the options in used in step 3.2:
```
rustup default nightly
```
3. PyO3 uses procedural macros, which have to expanded:
    1. First install cargo-expand (if not already installed):
    ```
    cargo install cargo-expand
    ```
    2. Create, save, and format the expanded code into `lib.rs`. MAKE SURE TO HAVE RENAMED `lib.rs` PREVIOUSLY, SO IT'S NOT OVERWRITTEN.
    ```
    cargo rustc --profile=check -- -Zunpretty=expanded > expanded.rs; rustfmt expanded.rs
    ```
4. Fix issues in the code, if any. Note: it's weird that has to be done, check why.
5. Compile code, create package and install in virtual env (make sure the venv is activated):
```
maturin develop
```
6. Start debugging python application, get the pid of the relevant process using `os.getpid()`.
7. Start debugging rust code (the expanded `lib.rs`). This will ask for a process to attach to. Copy the pid of the python process.
8. Once finished, switch back to stable:
```
rustup default stable
```
9. ... And rename `lib.rs` to continue development
```
mv src/lib_unexpanded.rs src/lib.rs
```