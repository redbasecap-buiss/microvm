/// Trap handling for RISC-V M-mode

use core::sync::atomic::{AtomicU64, Ordering};

const CLINT_BASE: usize = 0x0200_0000;
const CLINT_MTIMECMP: usize = CLINT_BASE + 0x4000;
const CLINT_MTIME: usize = CLINT_BASE + 0xBFF8;
const TIMER_INTERVAL: u64 = 10_000_000;

static TICKS: AtomicU64 = AtomicU64::new(0);

fn timer_set_next() {
    unsafe {
        let mtime = (CLINT_MTIME as *const u64).read_volatile();
        (CLINT_MTIMECMP as *mut u64).write_volatile(mtime + TIMER_INTERVAL);
    }
}

pub fn init() {
    timer_set_next();

    unsafe {
        // Enable MTIE (bit 7)
        let mut mie: u64;
        core::arch::asm!("csrr {}, mie", out(reg) mie);
        mie |= 1 << 7;
        core::arch::asm!("csrw mie, {}", in(reg) mie);

        // Enable global interrupts MIE (bit 3)
        let mut mstatus: u64;
        core::arch::asm!("csrr {}, mstatus", out(reg) mstatus);
        mstatus |= 1 << 3;
        core::arch::asm!("csrw mstatus, {}", in(reg) mstatus);

        // Set trap vector
        let handler = trap_entry as usize;
        core::arch::asm!("csrw mtvec, {}", in(reg) handler);
    }
}

pub fn tick_count() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn trap_handler_rust(mcause: u64, mepc: u64) -> u64 {
    let is_interrupt = mcause >> 63 != 0;
    let code = mcause & !(1u64 << 63);

    if is_interrupt && code == 7 {
        // Machine timer interrupt
        TICKS.fetch_add(1, Ordering::Relaxed);
        timer_set_next();
    }

    mepc
}

// Assembly trap entry
core::arch::global_asm!(
    r#"
.align 4
.globl trap_entry
trap_entry:
    addi sp, sp, -128
    sd ra,  0(sp)
    sd t0,  8(sp)
    sd t1, 16(sp)
    sd t2, 24(sp)
    sd a0, 32(sp)
    sd a1, 40(sp)
    sd a2, 48(sp)
    sd a3, 56(sp)
    sd a4, 64(sp)
    sd a5, 72(sp)
    sd a6, 80(sp)
    sd a7, 88(sp)

    csrr a0, mcause
    csrr a1, mepc
    call trap_handler_rust
    csrw mepc, a0

    ld ra,  0(sp)
    ld t0,  8(sp)
    ld t1, 16(sp)
    ld t2, 24(sp)
    ld a0, 32(sp)
    ld a1, 40(sp)
    ld a2, 48(sp)
    ld a3, 56(sp)
    ld a4, 64(sp)
    ld a5, 72(sp)
    ld a6, 80(sp)
    ld a7, 88(sp)
    addi sp, sp, 128
    mret
"#
);
