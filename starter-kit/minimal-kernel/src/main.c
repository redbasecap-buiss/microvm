// main.c — MyOS Kernel Main
// A minimal RISC-V kernel demonstrating UART, interrupts, paging, and scheduling.

#include "uart.h"
#include "trap.h"
#include "mm.h"
#include "sched.h"

// Defined by linker script
extern char _heap_start[];
extern char _heap_end[];

// ─── Demo Tasks ──────────────────────────────────────────

static void task_a(void) {
    for (int i = 0; i < 5; i++) {
        uart_printf("[Task A] Hello! tick=%d, pid=%d, iteration=%d\n",
                     (int)get_tick_count(), current_pid(), i);
        // Busy wait to let timer fire
        for (volatile int j = 0; j < 1000000; j++);
        yield();
    }
    uart_puts("[Task A] done.\n");
}

static void task_b(void) {
    for (int i = 0; i < 5; i++) {
        uart_printf("[Task B] World! tick=%d, pid=%d, iteration=%d\n",
                     (int)get_tick_count(), current_pid(), i);
        for (volatile int j = 0; j < 1000000; j++);
        yield();
    }
    uart_puts("[Task B] done.\n");
}

static void task_c(void) {
    uart_printf("[Task C] I'm task %d. Quick syscall demo:\n", current_pid());

    // Demonstrate ecall (write syscall)
    const char msg[] = "[Task C] Written via syscall!\n";
    register uint64_t a7 asm("a7") = 1;           // SYS_WRITE
    register uint64_t a0 asm("a0") = 1;           // fd (ignored)
    register uint64_t a1 asm("a1") = (uint64_t)msg;
    register uint64_t a2 asm("a2") = sizeof(msg) - 1;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7));

    uart_puts("[Task C] done.\n");
}

// ─── Kernel Entry Point ──────────────────────────────────

void kernel_main(void) {
    uart_init();

    uart_puts("\n");
    uart_puts("  ╔═══════════════════════════════════╗\n");
    uart_puts("  ║     🖥️  MyOS v0.1 on RISC-V       ║\n");
    uart_puts("  ║     Hello from my OS!              ║\n");
    uart_puts("  ╚═══════════════════════════════════╝\n");
    uart_puts("\n");

    // Initialize memory manager
    mm_init((uint64_t)_heap_start, (uint64_t)_heap_end);

    // Set up Sv39 page tables
    vm_init();

    // Initialize scheduler
    sched_init();

    // Create demo tasks
    task_create(task_a, "Task A");
    task_create(task_b, "Task B");
    task_create(task_c, "Task C");

    // Enable timer interrupts
    trap_init();

    uart_puts("[kernel] starting scheduler...\n\n");

    // Start scheduling — this doesn't return
    schedule();

    // Should never reach here
    uart_puts("[kernel] all tasks completed. Halting.\n");
    while (1) asm volatile("wfi");
}
