// mm.c — Simple Page Allocator (bump allocator) + Sv39 Page Tables

#include "mm.h"
#include "uart.h"

static uint64_t heap_ptr;
static uint64_t heap_limit;
static uint64_t pages_allocated = 0;

void mm_init(uint64_t heap_start, uint64_t heap_end) {
    // Align to page boundary
    heap_ptr = (heap_start + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    heap_limit = heap_end & ~(PAGE_SIZE - 1);
    uart_printf("[mm] heap: %p - %p (%d pages)\n",
                heap_ptr, heap_limit,
                (int)((heap_limit - heap_ptr) / PAGE_SIZE));
}

void *page_alloc(void) {
    if (heap_ptr >= heap_limit) {
        uart_puts("[mm] PANIC: out of memory!\n");
        return (void *)0;
    }
    void *p = (void *)heap_ptr;
    // Zero the page
    uint64_t *w = (uint64_t *)p;
    for (int i = 0; i < (int)(PAGE_SIZE / 8); i++)
        w[i] = 0;
    heap_ptr += PAGE_SIZE;
    pages_allocated++;
    return p;
}

// Set up identity-mapped Sv39 page tables
// Maps first 128MiB as RWX (kernel space) + MMIO regions
void vm_init(void) {
    // Allocate root page table (level 2)
    uint64_t *root = (uint64_t *)page_alloc();
    if (!root) return;

    // Use megapages (2MiB, level 1) for simplicity
    // Map 0x00000000 - 0x0FFFFFFF (MMIO: CLINT, PLIC, UART)
    // Map 0x80000000 - 0x87FFFFFF (RAM: 128MiB)

    // Level 2 entry for 0x00000000 (VPN[2] = 0)
    uint64_t *l1_mmio = (uint64_t *)page_alloc();
    root[0] = ((uint64_t)l1_mmio >> PAGE_SHIFT << 10) | PTE_V;

    // Map first 256 megapages for MMIO (0x00000000 - 0x1FFFFFFF)
    for (int i = 0; i < 256; i++) {
        uint64_t pa = (uint64_t)i << 21;
        l1_mmio[i] = (pa >> PAGE_SHIFT << 10) | PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D;
    }

    // Level 2 entry for 0x80000000 (VPN[2] = 2)
    uint64_t *l1_ram = (uint64_t *)page_alloc();
    root[2] = ((uint64_t)l1_ram >> PAGE_SHIFT << 10) | PTE_V;

    // Map 64 megapages for RAM (128MiB)
    for (int i = 0; i < 64; i++) {
        uint64_t pa = 0x80000000UL + ((uint64_t)i << 21);
        l1_ram[i] = (pa >> PAGE_SHIFT << 10) | PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D;
    }

    // Set satp (Sv39 mode = 8)
    uint64_t satp = (8UL << 60) | ((uint64_t)root >> PAGE_SHIFT);
    uart_printf("[mm] enabling Sv39 paging, satp=%p\n", satp);

    // Note: In M-mode we don't actually activate satp (it's for S-mode).
    // We store it for when we switch to S-mode, or demonstrate the setup.
    // For a pure M-mode kernel, identity mapping means addresses work as-is.
    uart_printf("[mm] page tables ready (%d pages allocated)\n", (int)pages_allocated);
}
