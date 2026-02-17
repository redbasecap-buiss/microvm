# Tutorial 02: Interrupts — Timer Ticks

> ⏱️ Time: 15 minutes

Your kernel prints text, but it can't *react* to anything. Interrupts change that — they let the hardware poke your kernel when something happens.

## What Are Interrupts?

An interrupt is the CPU saying: "Stop what you're doing, something happened." On RISC-V, the main interrupt sources are:

| Source | CSR Bit | Description |
|--------|---------|-------------|
| Timer | MTIE (bit 7) | CLINT timer fired |
| Software | MSIE (bit 3) | Inter-processor interrupt |
| External | MEIE (bit 11) | PLIC device interrupt |

## The CLINT (Core Local Interruptor)

microvm emulates a CLINT at `0x02000000`:

| Address | Register | Description |
|---------|----------|-------------|
| `0x0200_4000` | `mtimecmp` | Timer fires when `mtime >= mtimecmp` |
| `0x0200_BFF8` | `mtime` | Free-running counter |

## Step 1: Set Up the Trap Vector

The CPU needs to know where to jump when an interrupt fires:

```c
// In boot.S — the trap entry saves all registers, calls C, restores them
.align 4
trap_entry:
    addi sp, sp, -256
    sd ra, 0(sp)
    sd t0, 8(sp)
    // ... save all registers ...

    csrr a0, mcause     // What happened?
    csrr a1, mepc       // Where were we?
    call trap_handler    // C function

    csrw mepc, a0       // Maybe update return address
    // ... restore all registers ...
    addi sp, sp, 256
    mret                 // Return from trap
```

Register `mtvec` (Machine Trap Vector) tells the CPU where `trap_entry` is:

```asm
la t0, trap_entry
csrw mtvec, t0
```

## Step 2: Program the Timer

```c
#define CLINT_MTIMECMP  (*(volatile uint64_t *)0x02004000)
#define CLINT_MTIME     (*(volatile uint64_t *)0x0200BFF8)
#define INTERVAL        10000000  // ~100ms at 100MHz

void timer_set_next(void) {
    CLINT_MTIMECMP = CLINT_MTIME + INTERVAL;
}
```

## Step 3: Enable Interrupts

Two CSR bits need to be set:
1. `mie.MTIE` (bit 7) — enable timer interrupt specifically
2. `mstatus.MIE` (bit 3) — enable interrupts globally

```c
void trap_init(void) {
    timer_set_next();

    uint64_t mie;
    asm volatile("csrr %0, mie" : "=r"(mie));
    mie |= (1 << 7);
    asm volatile("csrw mie, %0" :: "r"(mie));

    uint64_t mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (1 << 3);
    asm volatile("csrw mstatus, %0" :: "r"(mstatus));
}
```

## Step 4: Handle the Interrupt

```c
static uint64_t tick_count = 0;

uint64_t trap_handler(uint64_t mcause, uint64_t mepc) {
    if (mcause & (1UL << 63)) {
        // Interrupt (top bit set)
        uint64_t code = mcause & ~(1UL << 63);
        if (code == 7) {  // Machine timer
            tick_count++;
            uart_printf("[tick %d]\n", (int)tick_count);
            timer_set_next();
        }
    }
    return mepc;
}
```

**Why return `mepc`?** After handling the interrupt, `mret` jumps back to wherever the CPU was interrupted. For exceptions (like `ecall`), we return `mepc + 4` to skip the faulting instruction.

## Run It

```bash
make && make run
```

You should see periodic tick messages appearing!

## Key Takeaways

- `mtvec` → where to jump on interrupt
- `mie` → which interrupts are enabled
- `mstatus.MIE` → global interrupt enable
- `mcause` bit 63 → interrupt (1) vs exception (0)
- Always reprogram `mtimecmp` after each timer fire

## What's Next?

→ [Tutorial 03: Virtual Memory](03-virtual-memory.md) — Set up Sv39 page tables
