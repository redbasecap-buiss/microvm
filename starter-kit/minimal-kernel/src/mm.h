#ifndef MM_H
#define MM_H

#include <stdint.h>
#include <stddef.h>

// Sv39 page table constants
#define PAGE_SIZE    4096
#define PAGE_SHIFT   12
#define PTE_V        (1 << 0)  // Valid
#define PTE_R        (1 << 1)  // Read
#define PTE_W        (1 << 2)  // Write
#define PTE_X        (1 << 3)  // Execute
#define PTE_U        (1 << 4)  // User
#define PTE_A        (1 << 6)  // Accessed
#define PTE_D        (1 << 7)  // Dirty

void mm_init(uint64_t heap_start, uint64_t heap_end);
void *page_alloc(void);
void vm_init(void);

#endif
