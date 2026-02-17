# Tutorial 01: Hello World — Your First Boot

> ⏱️ Time: 10 minutes

In this tutorial you'll write a bare-metal RISC-V program that prints "Hello from my OS!" to the serial console.

## How It Works

When microvm starts your kernel, it loads the binary at address `0x80000000` and jumps to it. There's no operating system, no libc — just your code and the hardware.

The simplest way to output text is through the **UART** (Universal Asynchronous Receiver/Transmitter). microvm emulates a 16550-compatible UART at address `0x10000000`.

## Step 1: The Entry Point (`boot.S`)

Every kernel needs an assembly entry point to set up the stack before jumping to C:

```asm
.section .text.entry
.globl _start

_start:
    csrw mie, zero          # Disable interrupts
    csrr t0, mhartid        # Read hart (core) ID
    bnez t0, park            # Only hart 0 boots

    la   sp, _stack_top      # Set up stack pointer
    call kernel_main         # Jump to C!

park:
    wfi                      # Wait for interrupt (halt)
    j    park
```

**Key concepts:**
- `mhartid` — each CPU core has an ID; we only want core 0 to boot
- `_stack_top` — defined in the linker script
- `wfi` — puts the CPU to sleep (saves power)

## Step 2: UART Driver (`uart.c`)

Writing to the UART is simple — write a byte to address `0x10000000`:

```c
#define UART_BASE 0x10000000UL
#define UART_LSR  5
#define UART_LSR_TX_EMPTY 0x20

static volatile uint8_t *const uart = (volatile uint8_t *)UART_BASE;

void uart_putchar(char c) {
    // Wait until transmit buffer is empty
    while ((uart[UART_LSR] & UART_LSR_TX_EMPTY) == 0)
        ;
    uart[0] = (uint8_t)c;
}

void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putchar('\r');  // Serial terminals need \r\n
        uart_putchar(*s++);
    }
}
```

**Why `volatile`?** The compiler might optimize away repeated reads/writes to the same address. `volatile` tells it "this is hardware — every access matters."

## Step 3: Kernel Main (`main.c`)

```c
void kernel_main(void) {
    uart_init();
    uart_puts("Hello from my OS!\n");

    // Halt
    while (1) asm volatile("wfi");
}
```

## Step 4: Linker Script

The linker script tells the compiler where to place code in memory:

```ld
ENTRY(_start)
SECTIONS {
    . = 0x80000000;          /* microvm loads kernel here */
    .text : { *(.text.entry) *(.text*) }
    .data : { *(.data*) }
    .bss  : { *(.bss*) }
    . += 0x4000;
    _stack_top = .;           /* 16 KiB stack */
}
```

## Step 5: Build & Run

```bash
riscv64-elf-gcc -ffreestanding -nostdlib -march=rv64gc -mabi=lp64d \
    -T linker.ld -o kernel.elf boot.S uart.c main.c
riscv64-elf-objcopy -O binary kernel.elf kernel.bin
microvm run --kernel kernel.bin --load-addr 0x80000000
```

You should see:
```
Hello from my OS!
```

🎉 **Congratulations!** You just booted your own operating system.

## What's Next?

→ [Tutorial 02: Interrupts](02-interrupts.md) — Handle timer interrupts
