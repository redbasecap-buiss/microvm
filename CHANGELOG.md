# Changelog

## v0.2.0 — OS Dev Playground (2026-02-18)

### ✨ New Features

- **Starter Kit: C Kernel** (`starter-kit/minimal-kernel/`)
  - Complete RISC-V kernel in <500 lines of C
  - 16550 UART driver with printf-lite
  - Timer interrupt handling via CLINT
  - Sv39 virtual memory page table setup
  - Round-robin scheduler with 3 demo tasks
  - 4 syscalls: write, exit, yield, getpid
  - Cross-compilation with riscv64-elf-gcc

- **Starter Kit: Rust Kernel** (`starter-kit/rust-kernel/`)
  - `#![no_std]` `#![no_main]` RISC-V kernel
  - UART driver, timer interrupts
  - Assembly boot stub and trap entry

- **6 Tutorials** (`docs/tutorials/`)
  - 01: Hello World — UART output, first boot
  - 02: Interrupts — Timer interrupt handling
  - 03: Virtual Memory — Sv39 page tables
  - 04: Userspace — Privilege modes, syscalls
  - 05: Scheduler — Round-robin multitasking
  - 06: Rust Kernel — OS dev in Rust

- **Project Scaffolding** (`microvm-init.sh`)
  - One-command project creation from templates
  - Supports C and Rust templates

- **Standalone Starter Kit Repo**
  - [microvm-starter-kit](https://github.com/redbasecap-buiss/microvm-starter-kit)
  - Fork/clone independently from the emulator

### 📝 Changes

- README rebranded as "OS Development Playground"
- Updated roadmap

## v0.1.0 — Foundation (2025-01-15)

- RV64GC CPU emulation (I, M, A, C extensions)
- Machine/Supervisor/User privilege modes
- Sv39 MMU with 3-level page table walking
- 16550 UART with terminal I/O
- CLINT (timer + software interrupts)
- PLIC (external interrupts)
- Auto-generated Device Tree Blob
- Boot ROM trampoline
- CLI with clap
