#![no_std]
#![no_main]

mod uart;
mod trap;

use core::panic::PanicInfo;
use uart::Uart;

/// Entry point — called from assembly boot stub
#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    let uart = Uart::new(0x1000_0000);
    uart.init();

    uart.puts("\n");
    uart.puts("  ╔═══════════════════════════════════╗\n");
    uart.puts("  ║   🦀 RustOS v0.1 on RISC-V        ║\n");
    uart.puts("  ║   Hello from my Rust OS!           ║\n");
    uart.puts("  ╚═══════════════════════════════════╝\n");
    uart.puts("\n");

    // Enable timer interrupts
    trap::init();

    uart.puts("[kernel] timer interrupts enabled\n");
    uart.puts("[kernel] entering idle loop...\n\n");

    let mut tick = 0u64;
    loop {
        // Simple demo: print on timer ticks
        let new_tick = trap::tick_count();
        if new_tick > tick {
            tick = new_tick;
            uart.puts("[kernel] tick ");
            uart.put_dec(tick);
            uart.puts("\n");

            if tick >= 10 {
                uart.puts("\n[kernel] 10 ticks done. Halting.\n");
                break;
            }
        }
        // Wait for interrupt
        unsafe { core::arch::asm!("wfi") };
    }

    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let uart = Uart::new(0x1000_0000);
    uart.puts("PANIC: ");
    if let Some(loc) = info.location() {
        uart.puts(loc.file());
        uart.puts(":");
        uart.put_dec(loc.line() as u64);
    }
    uart.puts("\n");
    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}
