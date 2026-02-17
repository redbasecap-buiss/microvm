#ifndef TRAP_H
#define TRAP_H

#include <stdint.h>

// CLINT addresses (matches microvm memory map)
#define CLINT_BASE       0x02000000UL
#define CLINT_MTIMECMP   (CLINT_BASE + 0x4000)
#define CLINT_MTIME      (CLINT_BASE + 0xBFF8)

// Timer interval (~10M cycles ≈ 100ms at 100MHz)
#define TIMER_INTERVAL   10000000UL

// mcause values
#define CAUSE_MACHINE_TIMER    7
#define CAUSE_MACHINE_ECALL   11
#define CAUSE_INTERRUPT_BIT   (1UL << 63)

void trap_init(void);
uint64_t trap_handler(uint64_t mcause, uint64_t mepc, uint64_t mtval, void *frame);
uint64_t get_tick_count(void);

#endif
