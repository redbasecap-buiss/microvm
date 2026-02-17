# Tutorial 06: Rust Kernel — `#![no_std]` OS Development

> ⏱️ Time: 15 minutes

Everything we built in C, now in Rust. Rust's zero-cost abstractions and memory safety make it an excellent language for OS development.

## Why Rust for OS Dev?

- **No undefined behavior** (outside `unsafe`)
- **No null pointers, no data races** by default
- **Zero-cost abstractions** — as fast as C
- **Package manager** (cargo) — even for bare metal
- **Growing RISC-V ecosystem**

## Step 1: Project Setup

```bash
cargo new --lib rust-kernel
cd rust-kernel
rustup target add riscv64gc-unknown-none-elf
```

### `Cargo.toml`
```toml
[package]
name = "rust-kernel"
version = "0.1.0"
edition = "2021"

[profile.release]
opt-level = "s"
lto = true
panic = "abort"
```

### `.cargo/config.toml`
```toml
[build]
target = "riscv64gc-unknown-none-elf"

[target.riscv64gc-unknown-none-elf]
rustflags = ["-C", "link-arg=-Tlinker.ld", "-C", "link-arg=-nostartfiles"]
```

## Step 2: The `#![no_std]` Kernel

```rust
#![no_std]
#![no_main]

use core::panic::PanicInfo;

mod uart;

#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    let uart = uart::Uart::new(0x1000_0000);
    uart.init();
    uart.puts("Hello from Rust OS!\n");

    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop { unsafe { core::arch::asm!("wfi") }; }
}
```

**Key attributes:**
- `#![no_std]` — no standard library (no OS to provide it!)
- `#![no_main]` — no Rust runtime entry point
- `#[no_mangle]` — keep function name for linker
- `-> !` — function never returns (diverging)

## Step 3: UART in Rust

```rust
pub struct Uart { base: usize }

impl Uart {
    pub const fn new(base: usize) -> Self { Uart { base } }

    pub fn putchar(&self, c: u8) {
        unsafe {
            let lsr = (self.base + 5) as *const u8;
            while lsr.read_volatile() & 0x20 == 0 {}
            (self.base as *mut u8).write_volatile(c);
        }
    }

    pub fn puts(&self, s: &str) {
        for b in s.bytes() {
            if b == b'\n' { self.putchar(b'\r'); }
            self.putchar(b);
        }
    }
}
```

Note: `unsafe` is confined to hardware access. The API is safe.

## Step 4: Inline Assembly for CSRs

Rust has native inline assembly:

```rust
// Read a CSR
let mie: u64;
unsafe { core::arch::asm!("csrr {}, mie", out(reg) mie) };

// Write a CSR
unsafe { core::arch::asm!("csrw mie, {}", in(reg) mie | (1 << 7)) };
```

## Step 5: Global Assembly for Boot & Trap Entry

```rust
core::arch::global_asm!(r#"
.section .text.entry
.globl _start
_start:
    csrw mie, zero
    la sp, _stack_top
    call kernel_main
1:  wfi
    j 1b
"#);
```

## Step 6: Build & Run

```bash
cd starter-kit/rust-kernel
make
make run
```

## Rust vs C: Side by Side

| Aspect | C Kernel | Rust Kernel |
|--------|----------|-------------|
| Entry | `void kernel_main()` | `pub extern "C" fn kernel_main() -> !` |
| UART write | `uart[0] = c` | `ptr.write_volatile(c)` |
| Inline asm | `asm volatile(...)` | `core::arch::asm!(...)` |
| Safety | Your problem | Compiler helps |
| Build | `make` + cross-gcc | `cargo build` |

## Going Further

The starter kit Rust kernel includes:
- UART driver with number formatting
- Timer interrupt handling
- Assembly trap entry/exit

To add a scheduler, port the C version's `context_switch` using `global_asm!` and create task structures with Rust's type system.

## Resources

- [The Embedded Rust Book](https://docs.rust-embedded.org/book/)
- [Writing an OS in Rust](https://os.phil-opp.com/) (x86, but concepts apply)
- [riscv crate](https://crates.io/crates/riscv) — RISC-V register access
