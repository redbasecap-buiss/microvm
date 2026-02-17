/* linker.ld — RISC-V Linker Script for MyOS */

OUTPUT_ARCH(riscv)
ENTRY(_start)

BASE_ADDR = 0x80000000;

SECTIONS
{
    . = BASE_ADDR;

    .text : {
        *(.text.entry)    /* Entry point first */
        *(.text .text.*)
    }

    . = ALIGN(4096);
    .rodata : {
        *(.rodata .rodata.*)
    }

    . = ALIGN(4096);
    .data : {
        *(.data .data.*)
        *(.sdata .sdata.*)
    }

    . = ALIGN(4096);
    _bss_start = .;
    .bss : {
        *(.bss .bss.*)
        *(.sbss .sbss.*)
    }
    _bss_end = .;

    . = ALIGN(4096);
    _stack_bottom = .;
    . += 0x4000;          /* 16 KiB stack */
    _stack_top = .;

    . = ALIGN(4096);
    _heap_start = .;
    _heap_end = BASE_ADDR + 0x800000;  /* 8 MiB total */
}
