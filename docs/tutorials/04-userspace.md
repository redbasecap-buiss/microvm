# Tutorial 04: Userspace — From M-Mode to U-Mode

> ⏱️ Time: 20 minutes

Real operating systems don't run everything in machine mode. They drop to a lower privilege level and use **syscalls** to request kernel services.

## RISC-V Privilege Levels

```
Level 3: Machine Mode (M)  ← Full hardware access (firmware/bootloader)
Level 1: Supervisor Mode (S) ← OS kernel (with MMU)
Level 0: User Mode (U)      ← Applications (restricted)
```

Our kernel runs in M-mode. In this tutorial, we'll implement syscalls that user tasks use via the `ecall` instruction.

## How Syscalls Work

1. User code puts syscall number in `a7`, arguments in `a0`-`a6`
2. User code executes `ecall`
3. CPU traps to M-mode (jumps to `mtvec`)
4. Trap handler reads `mcause` → sees "environment call from M-mode" (code 11)
5. Handler dispatches based on `a7`, returns result in `a0`
6. `mret` returns to user code

## Step 1: Define Syscalls

We implement 4 essential syscalls:

| Number | Name | Args | Description |
|--------|------|------|-------------|
| 1 | `write` | fd, buf, len | Write bytes to UART |
| 2 | `exit` | code | Terminate task |
| 3 | `yield` | — | Give up CPU |
| 4 | `getpid` | — | Get task ID |

## Step 2: Syscall Handler

```c
void syscall_dispatch(void *frame) {
    trap_frame_t *f = (trap_frame_t *)frame;

    switch (f->a7) {
    case SYS_WRITE: {
        const char *buf = (const char *)f->a1;
        for (uint64_t i = 0; i < f->a2; i++)
            uart_putchar(buf[i]);
        f->a0 = f->a2;  // Return bytes written
        break;
    }
    case SYS_EXIT:
        // Mark current task dead, schedule next
        yield();
        break;
    case SYS_YIELD:
        yield();
        break;
    case SYS_GETPID:
        f->a0 = current_pid();
        break;
    }
}
```

## Step 3: Trap Handler Update

In the trap handler, detect `ecall` and advance `mepc`:

```c
uint64_t trap_handler(uint64_t mcause, uint64_t mepc, uint64_t mtval, void *frame) {
    if (mcause & (1UL << 63)) {
        // Interrupt — handle as before
    } else if (mcause == 11) {
        // Environment call from M-mode
        syscall_dispatch(frame);
        return mepc + 4;  // Skip past ecall instruction!
    }
    return mepc;
}
```

**Critical**: Return `mepc + 4` for ecalls! Otherwise the CPU re-executes `ecall` forever.

## Step 4: User Code Making Syscalls

```c
// In a user task:
static void task_demo(void) {
    const char msg[] = "Hello via syscall!\n";

    register uint64_t a7 asm("a7") = 1;           // SYS_WRITE
    register uint64_t a0 asm("a0") = 1;           // fd
    register uint64_t a1 asm("a1") = (uint64_t)msg;
    register uint64_t a2 asm("a2") = sizeof(msg) - 1;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7));

    // a0 now contains the return value (bytes written)
}
```

## The Trap Frame

The register layout saved on the stack must match exactly:

```
Offset   Register   Purpose
  0      ra         Return address
  8      t0         Temporary
 ...     ...
 32      a0         Arg 0 / return value
 40      a1         Arg 1
 48      a2         Arg 2
 ...     ...
 88      a7         Syscall number
```

## What's Next?

→ [Tutorial 05: Scheduler](05-scheduler.md) — Round-robin multitasking
