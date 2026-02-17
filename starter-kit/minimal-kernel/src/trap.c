// trap.c — Basic Trap Handler (timer, ecall)

#include "trap.h"
#include "uart.h"
#include "sched.h"
#include "syscall.h"

static volatile uint64_t tick_count = 0;

static void timer_set_next(void) {
    volatile uint64_t *mtime    = (volatile uint64_t *)CLINT_MTIME;
    volatile uint64_t *mtimecmp = (volatile uint64_t *)CLINT_MTIMECMP;
    *mtimecmp = *mtime + TIMER_INTERVAL;
}

void trap_init(void) {
    timer_set_next();

    // Enable machine timer interrupt (MIE.MTIE)
    uint64_t mie;
    asm volatile("csrr %0, mie" : "=r"(mie));
    mie |= (1 << 7);  // MTIE
    asm volatile("csrw mie, %0" :: "r"(mie));

    // Enable global interrupts (MSTATUS.MIE)
    uint64_t mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (1 << 3);  // MIE
    asm volatile("csrw mstatus, %0" :: "r"(mstatus));
}

uint64_t trap_handler(uint64_t mcause, uint64_t mepc, uint64_t mtval, void *frame) {
    (void)mtval;
    (void)frame;

    if (mcause & CAUSE_INTERRUPT_BIT) {
        // Interrupt
        uint64_t code = mcause & ~CAUSE_INTERRUPT_BIT;
        if (code == CAUSE_MACHINE_TIMER) {
            tick_count++;
            timer_set_next();
            schedule();  // Context switch on timer tick
        }
    } else {
        // Exception
        if (mcause == CAUSE_MACHINE_ECALL) {
            syscall_dispatch(frame);
            return mepc + 4;  // Skip ecall instruction
        }
        // Unknown exception — print and halt
        uart_printf("PANIC: exception cause=%x mepc=%p mtval=%p\n",
                     (unsigned int)mcause, mepc, mtval);
        while (1) asm volatile("wfi");
    }

    return mepc;
}

uint64_t get_tick_count(void) {
    return tick_count;
}
