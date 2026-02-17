# 🖥️ MyOS — Build Your First OS

A minimal RISC-V kernel in under 500 lines of C. Boots on [microvm](https://github.com/redbasecap-buiss/microvm).

## What It Does

- 🖨️ Prints "Hello from my OS!" via 16550 UART
- ⏰ Handles timer interrupts (CLINT)
- 📋 Schedules 3 tasks with round-robin scheduling
- 🔧 Implements 4 syscalls: `write`, `exit`, `yield`, `getpid`
- 🗺️ Sets up Sv39 virtual memory page tables
- 💀 Fits in <500 lines of C (excluding boot assembly)

## Prerequisites

You need a RISC-V cross-compiler:

```bash
# macOS
brew install riscv64-elf-gcc

# Ubuntu/Debian
sudo apt install gcc-riscv64-unknown-elf

# Arch
sudo pacman -S riscv64-elf-gcc
```

## Build & Run

```bash
# Build the kernel
make

# Run on microvm
make run
# or: microvm run --kernel kernel.bin --load-addr 0x80000000
```

## Expected Output

```
  ╔═══════════════════════════════════╗
  ║     🖥️  MyOS v0.1 on RISC-V       ║
  ║     Hello from my OS!              ║
  ╚═══════════════════════════════════╝

[mm] heap: 0x80006000 - 0x80800000 (2042 pages)
[mm] enabling Sv39 paging, satp=0x8000000080006
[mm] page tables ready (3 pages allocated)
[sched] created task 1: 'Task A'
[sched] created task 2: 'Task B'
[sched] created task 3: 'Task C'
[kernel] starting scheduler...

[Task A] Hello! tick=0, pid=1, iteration=0
[Task B] World! tick=1, pid=2, iteration=0
[Task C] I'm task 3. Quick syscall demo:
[Task C] Written via syscall!
[Task C] done.
[Task A] Hello! tick=2, pid=1, iteration=1
...
```

## File Structure

| File | Lines | Description |
|------|-------|-------------|
| `boot.S` | ~90 | Assembly entry: stack setup, trap vector, register save/restore |
| `main.c` | ~80 | Kernel entry, task definitions, boot sequence |
| `uart.c` | ~80 | 16550 UART driver with printf-lite |
| `trap.c` | ~55 | Timer interrupt handler, exception dispatch |
| `mm.c` | ~70 | Bump page allocator, Sv39 page table setup |
| `sched.c` | ~100 | Round-robin scheduler with context switching |
| `syscall.c` | ~50 | write, exit, yield, getpid syscall handlers |

## Next Steps

1. 📖 Follow the [tutorials](../../docs/tutorials/) to understand each component
2. 🔨 Add your own syscalls in `syscall.c`
3. 🎮 Create new tasks in `main.c`
4. 🗂️ Implement a simple filesystem
5. 🌐 Add VirtIO device drivers

## See Also

- [Tutorials](../../docs/tutorials/) — Step-by-step guides
- [Rust Kernel](../rust-kernel/) — Same thing in Rust
- [microvm](https://github.com/redbasecap-buiss/microvm) — The emulator
