# boot.S — RISC-V Assembly Entry Point for MyOS
# Sets up stack and jumps to kernel_main

.section .text.entry
.globl _start

_start:
    # Disable interrupts
    csrw    mie, zero

    # Set up stack pointer
    # Each hart gets 4KB stack, hart 0 at top of stack area
    csrr    t0, mhartid
    bnez    t0, _park          # Only hart 0 boots

    la      sp, _stack_top

    # Clear BSS
    la      t0, _bss_start
    la      t1, _bss_end
1:
    bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:

    # Set up trap vector
    la      t0, trap_entry
    csrw    mtvec, t0

    # Jump to C kernel
    call    kernel_main

    # If kernel returns, park
_park:
    wfi
    j       _park

# Trap entry point (aligned)
.align 4
.globl trap_entry
trap_entry:
    # Save registers on stack
    addi    sp, sp, -256

    sd      ra,   0(sp)
    sd      t0,   8(sp)
    sd      t1,  16(sp)
    sd      t2,  24(sp)
    sd      a0,  32(sp)
    sd      a1,  40(sp)
    sd      a2,  48(sp)
    sd      a3,  56(sp)
    sd      a4,  64(sp)
    sd      a5,  72(sp)
    sd      a6,  80(sp)
    sd      a7,  88(sp)
    sd      t3,  96(sp)
    sd      t4, 104(sp)
    sd      t5, 112(sp)
    sd      t6, 120(sp)
    sd      s0, 128(sp)
    sd      s1, 136(sp)

    # Call C trap handler
    csrr    a0, mcause
    csrr    a1, mepc
    csrr    a2, mtval
    mv      a3, sp
    call    trap_handler

    # Return value = new mepc
    csrw    mepc, a0

    # Restore registers
    ld      ra,   0(sp)
    ld      t0,   8(sp)
    ld      t1,  16(sp)
    ld      t2,  24(sp)
    ld      a0,  32(sp)
    ld      a1,  40(sp)
    ld      a2,  48(sp)
    ld      a3,  56(sp)
    ld      a4,  64(sp)
    ld      a5,  72(sp)
    ld      a6,  80(sp)
    ld      a7,  88(sp)
    ld      t3,  96(sp)
    ld      t4, 104(sp)
    ld      t5, 112(sp)
    ld      t6, 120(sp)
    ld      s0, 128(sp)
    ld      s1, 136(sp)

    addi    sp, sp, 256
    mret
