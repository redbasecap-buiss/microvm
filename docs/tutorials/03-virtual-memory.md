# Tutorial 03: Virtual Memory — Sv39 Page Tables

> ⏱️ Time: 20 minutes

Virtual memory is the foundation of process isolation. On RISC-V, the standard paging scheme is **Sv39** — 39-bit virtual addresses, 3-level page tables.

## Sv39 Address Translation

A 39-bit virtual address is split into:

```
 38      30 29     21 20     12 11        0
┌─────────┬─────────┬─────────┬───────────┐
│ VPN[2]  │ VPN[1]  │ VPN[0]  │  Offset   │
│ 9 bits  │ 9 bits  │ 9 bits  │  12 bits  │
└─────────┴─────────┴─────────┴───────────┘
```

Each VPN (Virtual Page Number) indexes into a page table. Each page table has 512 entries (512 × 8 bytes = 4096 bytes = 1 page).

## Page Table Entries (PTE)

```
 63    54 53     10 9  8 7 6 5 4 3 2 1 0
┌────────┬─────────┬─┬──┬─┬─┬─┬─┬─┬─┬─┐
│Reserved│  PPN    │  │DA│ │X│W│R│V│ │ │
└────────┴─────────┴─┴──┴─┴─┴─┴─┴─┴─┴─┘

V = Valid      R = Read      W = Write     X = Execute
U = User       A = Accessed  D = Dirty
```

**Megapages**: If a level-1 PTE has RWX bits set, it maps a 2 MiB region directly (no level-0 walk needed). This is what we'll use for simplicity.

## Step 1: Page Allocator

Before creating page tables, we need a way to allocate pages:

```c
static uint64_t heap_ptr;

void mm_init(uint64_t start, uint64_t end) {
    heap_ptr = (start + 4095) & ~4095;  // Align up
}

void *page_alloc(void) {
    void *p = (void *)heap_ptr;
    // Zero the page
    for (int i = 0; i < 512; i++)
        ((uint64_t *)p)[i] = 0;
    heap_ptr += 4096;
    return p;
}
```

## Step 2: Build Page Tables

For a kernel, identity mapping (virtual = physical) is simplest:

```c
void vm_init(void) {
    uint64_t *root = page_alloc();  // Level-2 table

    // Map MMIO region (0x00000000 - 0x1FFFFFFF)
    uint64_t *l1_mmio = page_alloc();
    root[0] = ((uint64_t)l1_mmio >> 12 << 10) | PTE_V;

    for (int i = 0; i < 256; i++) {
        uint64_t pa = (uint64_t)i << 21;  // 2 MiB pages
        l1_mmio[i] = (pa >> 12 << 10) | PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D;
    }

    // Map RAM (0x80000000 - 0x87FFFFFF)
    uint64_t *l1_ram = page_alloc();
    root[2] = ((uint64_t)l1_ram >> 12 << 10) | PTE_V;

    for (int i = 0; i < 64; i++) {
        uint64_t pa = 0x80000000 + ((uint64_t)i << 21);
        l1_ram[i] = (pa >> 12 << 10) | PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D;
    }

    // Set satp (mode 8 = Sv39, PPN of root table)
    uint64_t satp = (8UL << 60) | ((uint64_t)root >> 12);
    asm volatile("csrw satp, %0" :: "r"(satp));
    asm volatile("sfence.vma");  // Flush TLB
}
```

## Understanding the Mapping

```
Virtual Address           Physical Address
0x0000_0000 ──────────── 0x0000_0000  (UART, CLINT, PLIC)
    ...                      ...
0x1FFF_FFFF ──────────── 0x1FFF_FFFF

0x8000_0000 ──────────── 0x8000_0000  (RAM)
    ...                      ...
0x87FF_FFFF ──────────── 0x87FF_FFFF
```

With identity mapping, addresses don't change — but the MMU is active and will fault on unmapped addresses, which is the foundation of memory protection.

## The `satp` Register

```
 63 60 59    44 43                  0
┌──────┬───────┬─────────────────────┐
│ Mode │ ASID  │        PPN          │
│  4   │  16   │        44           │
└──────┴───────┴─────────────────────┘

Mode 0 = No translation (bare)
Mode 8 = Sv39
Mode 9 = Sv48
```

## Important Notes

- **M-mode ignores `satp`** — page translation only applies in S-mode and U-mode
- Our starter kit kernel runs in M-mode, so we set up the tables but they're "ready for use"
- When you implement S-mode (Tutorial 04), the tables become active
- Always `sfence.vma` after changing page tables (flushes TLB)

## What's Next?

→ [Tutorial 04: Userspace](04-userspace.md) — Drop from M-mode to U-mode with syscalls
