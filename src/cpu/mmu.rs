use super::csr::{self, CsrFile};
use super::PrivilegeMode;
use crate::memory::Bus;

#[derive(Debug, Clone, Copy)]
pub enum AccessType {
    Read,
    Write,
    Execute,
}

// PTE bits
const PTE_V: u64 = 1 << 0;
const PTE_R: u64 = 1 << 1;
const PTE_W: u64 = 1 << 2;
const PTE_X: u64 = 1 << 3;
const PTE_U: u64 = 1 << 4;
const PTE_A: u64 = 1 << 6;
const PTE_D: u64 = 1 << 7;

// Svnapot: Naturally Aligned Power-of-Two pages (PTE bit 63)
const PTE_N: u64 = 1 << 63;

// Svpbmt: Page-Based Memory Types (PTE bits 62:61)
const PTE_PBMT_MASK: u64 = 0x3 << 61;
#[allow(dead_code)]
const PTE_PBMT_PMA: u64 = 0 << 61; // Use PMA attributes (default)
#[allow(dead_code)]
const PTE_PBMT_NC: u64 = 1 << 61; // Non-cacheable, idempotent (normal memory, no cache)
#[allow(dead_code)]
const PTE_PBMT_IO: u64 = 2 << 61; // Non-cacheable, non-idempotent (I/O)
const PTE_PBMT_RSVD: u64 = 3 << 61; // Reserved — causes page fault

/// TLB entry: cached virtual-to-physical page mapping
#[derive(Clone, Copy)]
struct TlbEntry {
    /// Virtual page number (vaddr >> 12)
    vpn: u64,
    /// Physical page number (paddr >> 12)
    ppn: u64,
    /// Page size shift (12 for 4K, 21 for 2M, 30 for 1G, 39 for 512G)
    page_shift: u8,
    /// PTE permission bits (R/W/X/U)
    pte_flags: u64,
    /// SATP value when this entry was created (for invalidation)
    satp: u64,
    /// Valid flag
    valid: bool,
}

impl Default for TlbEntry {
    fn default() -> Self {
        Self {
            vpn: 0,
            ppn: 0,
            page_shift: 12,
            pte_flags: 0,
            satp: 0,
            valid: false,
        }
    }
}

/// Number of TLB entries (must be power of 2)
const TLB_SIZE: usize = 256;

/// Sv39/Sv48 MMU with TLB — multi-level page table translation with A/D bit management
pub struct Mmu {
    /// Direct-mapped TLB cache
    tlb: Box<[TlbEntry; TLB_SIZE]>,
    /// TLB hit counter (for diagnostics)
    pub tlb_hits: u64,
    /// TLB miss counter (for diagnostics)
    pub tlb_misses: u64,
}

impl Default for Mmu {
    fn default() -> Self {
        Self::new()
    }
}

impl Mmu {
    pub fn new() -> Self {
        Self {
            tlb: Box::new([TlbEntry::default(); TLB_SIZE]),
            tlb_hits: 0,
            tlb_misses: 0,
        }
    }

    /// Flush the entire TLB (called on SFENCE.VMA, SINVAL.VMA, SATP writes)
    pub fn flush_tlb(&mut self) {
        for entry in self.tlb.iter_mut() {
            entry.valid = false;
        }
    }

    /// Flush TLB entries matching a specific virtual address
    pub fn flush_tlb_vaddr(&mut self, vaddr: u64) {
        let vpn = vaddr >> 12;
        // Check all entries since superpages may match different indices
        for entry in self.tlb.iter_mut() {
            if entry.valid {
                let mask = (1u64 << (entry.page_shift - 12)) - 1;
                if (entry.vpn & !mask) == (vpn & !mask) {
                    entry.valid = false;
                }
            }
        }
    }

    /// TLB index from virtual page number
    fn tlb_index(vpn: u64) -> usize {
        (vpn as usize) & (TLB_SIZE - 1)
    }

    /// Look up address in TLB
    fn tlb_lookup(
        &self,
        vaddr: u64,
        access: AccessType,
        mode: PrivilegeMode,
        csrs: &CsrFile,
    ) -> Option<u64> {
        let vpn = vaddr >> 12;
        let idx = Self::tlb_index(vpn);
        let entry = &self.tlb[idx];

        if !entry.valid {
            return None;
        }

        // Check SATP matches
        let satp = csrs.read(csr::SATP);
        if entry.satp != satp {
            return None;
        }

        // Check VPN matches (accounting for superpage size)
        let page_shift = entry.page_shift as u64;
        let vpn_mask = !((1u64 << (page_shift - 12)) - 1);
        if (entry.vpn & vpn_mask) != (vpn & vpn_mask) {
            return None;
        }

        // Check permissions
        if self
            .check_leaf_permissions(access, mode, entry.pte_flags, csrs)
            .is_err()
        {
            return None;
        }

        // For writes, check that dirty bit was already set (we only cache entries with A set,
        // and we need D for writes)
        if matches!(access, AccessType::Write) && entry.pte_flags & PTE_D == 0 {
            return None; // Force page walk to set D bit
        }

        // Construct physical address
        let offset_mask = (1u64 << page_shift) - 1;
        let phys = (entry.ppn << 12) | (vaddr & offset_mask);
        Some(phys)
    }

    /// Insert an entry into the TLB
    fn tlb_insert(&mut self, vaddr: u64, ppn: u64, page_shift: u8, pte_flags: u64, satp: u64) {
        let vpn = vaddr >> 12;
        let idx = Self::tlb_index(vpn);
        self.tlb[idx] = TlbEntry {
            vpn,
            ppn,
            page_shift,
            pte_flags,
            satp,
            valid: true,
        };
    }

    /// Translate virtual address to physical address.
    /// Returns Ok(physical_addr) or Err(exception_cause).
    /// Supports Sv39 (3-level) and Sv48 (4-level) page table walks.
    /// Sets Accessed and Dirty bits on page table entries as required by the spec.
    pub fn translate(
        &mut self,
        vaddr: u64,
        access: AccessType,
        mode: PrivilegeMode,
        csrs: &CsrFile,
        bus: &mut Bus,
    ) -> Result<u64, u64> {
        let satp = csrs.read(csr::SATP);
        let satp_mode = satp >> 60;

        // If bare mode or M-mode, no translation — but still check PMP
        if satp_mode == 0 || mode == PrivilegeMode::Machine {
            if !csrs.pmp_check(vaddr, 1, access, mode) {
                return Err(self.access_fault(access));
            }
            return Ok(vaddr);
        }

        // TLB lookup
        if let Some(phys) = self.tlb_lookup(vaddr, access, mode, csrs) {
            self.tlb_hits += 1;
            // PMP check still needed on TLB hits
            if !csrs.pmp_check(phys, 1, access, mode) {
                return Err(self.access_fault(access));
            }
            return Ok(phys);
        }
        self.tlb_misses += 1;

        let levels = match satp_mode {
            8 => 3,                // Sv39
            9 => 4,                // Sv48
            10 => 5,               // Sv57
            _ => return Ok(vaddr), // Unsupported mode, treat as bare
        };

        let ppn = satp & 0xFFF_FFFF_FFFF; // 44 bits
        let page_offset = vaddr & 0xFFF;

        // Extract VPN fields (each 9 bits)
        let vpn: [u64; 5] = [
            (vaddr >> 12) & 0x1FF,
            (vaddr >> 21) & 0x1FF,
            (vaddr >> 30) & 0x1FF,
            (vaddr >> 39) & 0x1FF,
            (vaddr >> 48) & 0x1FF,
        ];

        let mut a = ppn << 12;

        for level in (0..levels).rev() {
            let pte_addr = a + vpn[level] * 8;
            let pte = bus.read64(pte_addr);

            if pte & PTE_V == 0 {
                return Err(self.page_fault(access));
            }

            let r = pte & PTE_R;
            let w = pte & PTE_W;
            let x = pte & PTE_X;

            if r == 0 && w == 0 && x == 0 {
                // Pointer to next level
                a = ((pte >> 10) & 0xFFF_FFFF_FFFF) << 12;
                continue;
            }

            // Leaf PTE found

            // Svpbmt: check reserved PBMT encoding (11 = reserved → page fault)
            if pte & PTE_PBMT_MASK == PTE_PBMT_RSVD {
                return Err(self.page_fault(access));
            }

            // Check permissions
            self.check_leaf_permissions(access, mode, pte, csrs)?;

            // Svnapot: handle N bit on leaf PTEs (level 0 only)
            // When N=1, low PPN bits encode NAPOT size: ppn[3:0]=0b0111 → 64KiB (16×4KiB)
            let napot = pte & PTE_N != 0;
            if napot && level != 0 {
                // N bit set on superpage → reserved, page fault
                return Err(self.page_fault(access));
            }

            // Superpage alignment check: lower PPN bits must be zero
            let misaligned = match level {
                4 => ((pte >> 10) & 0xFFFFFFFFF) != 0, // 256 TiB (Sv57 level 4)
                3 => ((pte >> 10) & 0x7FFFFFF) != 0,   // 512 GiB (Sv48 level 3)
                2 => ((pte >> 10) & 0x3FFFF) != 0,     // 1 GiB
                1 => ((pte >> 10) & 0x1FF) != 0,       // 2 MiB
                _ => false,
            };
            if misaligned {
                return Err(self.page_fault(access));
            }

            // Svnapot: validate NAPOT encoding on level-0 leaves
            // Only ppn[3:0]=0b0111 is defined (64KiB); other patterns are reserved
            if napot {
                let ppn_low4 = (pte >> 10) & 0xF;
                if ppn_low4 != 0b0111 {
                    return Err(self.page_fault(access));
                }
            }

            // Update A/D bits (hardware-managed, as Linux expects)
            let need_a = pte & PTE_A == 0;
            let need_d = matches!(access, AccessType::Write) && pte & PTE_D == 0;
            if need_a || need_d {
                let mut new_pte = pte | PTE_A;
                if need_d {
                    new_pte |= PTE_D;
                }
                bus.write64(pte_addr, new_pte);
            }

            // Construct physical address
            let (ppn_pte, page_shift) = if napot {
                // 64KiB NAPOT: ppn[3:0] are part of the offset, effective shift = 16
                let ppn_base = ((pte >> 10) & 0xFFF_FFFF_FFFF) & !0xF_u64; // clear low 4 bits
                (ppn_base, 16u8)
            } else {
                let ppn_raw = (pte >> 10) & 0xFFF_FFFF_FFFF;
                let shift = match level {
                    4 => 48u8,
                    3 => 39,
                    2 => 30,
                    1 => 21,
                    0 => 12,
                    _ => unreachable!(),
                };
                (ppn_raw, shift)
            };
            let offset_mask = (1u64 << page_shift) - 1;
            let phys = if napot {
                (ppn_pte << 12) | (vaddr & offset_mask)
            } else {
                match level {
                    4 => (ppn_pte & !0xFFFFFFFFF) << 12 | (vaddr & ((1u64 << 48) - 1)), // 256 TiB
                    3 => (ppn_pte & !0x7FFFFFF) << 12 | (vaddr & 0xFF_FFFF_FFFF),       // 512 GiB
                    2 => (ppn_pte & !0x3FFFF) << 12 | (vaddr & 0x3FFFFFFF),             // 1 GiB
                    1 => (ppn_pte & !0x1FF) << 12 | (vaddr & 0x1FFFFF),                 // 2 MiB
                    0 => (ppn_pte << 12) | page_offset,                                 // 4 KiB
                    _ => unreachable!(),
                }
            };

            // PMP check on the translated physical address
            if !csrs.pmp_check(phys, 1, access, mode) {
                return Err(self.access_fault(access));
            }

            // Cache in TLB (with updated A/D bits)
            let cached_flags = pte | PTE_A | if need_d { PTE_D } else { 0 };
            let base_ppn = (phys & !offset_mask) >> 12;
            self.tlb_insert(vaddr, base_ppn, page_shift, cached_flags, satp);

            return Ok(phys);
        }

        Err(self.page_fault(access))
    }

    /// Check leaf PTE permissions for given access type and privilege mode.
    fn check_leaf_permissions(
        &self,
        access: AccessType,
        mode: PrivilegeMode,
        pte: u64,
        csrs: &CsrFile,
    ) -> Result<(), u64> {
        let r = pte & PTE_R;
        let w = pte & PTE_W;
        let x = pte & PTE_X;
        let u = pte & PTE_U;

        match access {
            AccessType::Read => {
                let mstatus = csrs.read(csr::MSTATUS);
                let mxr = (mstatus >> 19) & 1;
                if r == 0 && !(mxr == 1 && x != 0) {
                    return Err(self.page_fault(access));
                }
            }
            AccessType::Write => {
                if w == 0 {
                    return Err(self.page_fault(access));
                }
            }
            AccessType::Execute => {
                if x == 0 {
                    return Err(self.page_fault(access));
                }
            }
        }

        // Check U-bit
        match mode {
            PrivilegeMode::User => {
                if u == 0 {
                    return Err(self.page_fault(access));
                }
            }
            PrivilegeMode::Supervisor => {
                if u != 0 {
                    let mstatus = csrs.read(csr::MSTATUS);
                    let sum = (mstatus >> 18) & 1;
                    if sum == 0 {
                        return Err(self.page_fault(access));
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn page_fault(&self, access: AccessType) -> u64 {
        match access {
            AccessType::Execute => 12,
            AccessType::Read => 13,
            AccessType::Write => 15,
        }
    }

    /// Access fault exceptions (distinct from page faults — used for PMP violations)
    fn access_fault(&self, access: AccessType) -> u64 {
        match access {
            AccessType::Execute => 1, // Instruction access fault
            AccessType::Read => 5,    // Load access fault
            AccessType::Write => 7,   // Store/AMO access fault
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::csr::{CsrFile, MSTATUS, SATP};
    use crate::memory::Bus;

    const DRAM_BASE: u64 = crate::memory::DRAM_BASE; // 0x8000_0000
    const RAM_SIZE: u64 = 128 * 1024 * 1024; // 128 MiB

    /// Helper: set up permissive PMP (allow all for S/U mode) and SATP.
    fn setup_with_mode(root_page_phys: u64, mode: u64) -> (Bus, CsrFile, Mmu) {
        let bus = Bus::new(RAM_SIZE);
        let mut csrs = CsrFile::new();
        let mmu = Mmu::new();
        // Set SATP
        let satp = (mode << 60) | (root_page_phys >> 12);
        csrs.write_raw(SATP, satp);
        // Set up PMP entry 0 as NAPOT covering all of address space (allow RWX)
        // Use write() which updates the actual pmpcfg/pmpaddr fields
        csrs.write(0x3B0, u64::MAX); // pmpaddr0
                                     // pmpcfg0 byte 0: A=NAPOT(3), R=1, W=1, X=1 → 0b00011111 = 0x1F
        csrs.write(0x3A0, 0x1F); // pmpcfg0
        (bus, csrs, mmu)
    }

    fn setup_sv39(root_page_phys: u64) -> (Bus, CsrFile, Mmu) {
        setup_with_mode(root_page_phys, 8)
    }

    fn setup_sv48(root_page_phys: u64) -> (Bus, CsrFile, Mmu) {
        setup_with_mode(root_page_phys, 9)
    }

    fn setup_sv57(root_page_phys: u64) -> (Bus, CsrFile, Mmu) {
        setup_with_mode(root_page_phys, 10)
    }

    /// Create a leaf PTE with given PPN and flags (V is always set).
    fn leaf_pte(ppn: u64, flags: u64) -> u64 {
        (ppn << 10) | PTE_V | flags
    }

    /// Create a non-leaf (pointer) PTE pointing to the given physical page table address.
    fn pointer_pte(next_pt_phys: u64) -> u64 {
        ((next_pt_phys >> 12) << 10) | PTE_V
    }

    // ======================== Sv39 4KiB page ========================

    #[test]
    fn test_sv39_4k_page_read() {
        // Root page table at DRAM_BASE + 0x1_0000
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000; // physical page to map

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        // Map vaddr 0x0000_0000_0040_0000 (VPN[2]=0, VPN[1]=2, VPN[0]=0)
        let vaddr: u64 = 0x0040_0000;

        // L2 (root): entry 0 → L1
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        // L1: entry 2 → L0
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // L0: entry 0 → leaf (R+W+A+D)
        let target_ppn = target_phys >> 12;
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_ppn, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        // Write a marker value at target physical address
        bus.write64(target_phys + 0x100, 0xDEAD_BEEF_CAFE_BABE);

        let result = mmu.translate(
            vaddr + 0x100,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys + 0x100));
    }

    #[test]
    fn test_sv39_4k_page_write_sets_dirty() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // No A/D bits set initially
        bus.write64(l0_pt + 0 * 8, leaf_pte(target_phys >> 12, PTE_R | PTE_W));

        let result = mmu.translate(
            vaddr,
            AccessType::Write,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys));

        // Check that A and D bits were set by hardware
        let pte_after = bus.read64(l0_pt + 0 * 8);
        assert_ne!(pte_after & PTE_A, 0, "A bit should be set");
        assert_ne!(pte_after & PTE_D, 0, "D bit should be set");
    }

    #[test]
    fn test_sv39_2m_superpage() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        // 2MiB superpage at physical 0x8020_0000 (aligned)
        let target_base = DRAM_BASE + 0x20_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        // vaddr = 0x0020_0000 → VPN[2]=0, VPN[1]=1, VPN[0]=0
        let vaddr: u64 = 0x0020_0000;

        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        // L1 entry 1: leaf (superpage 2MiB), PPN must have low 9 bits = 0
        let ppn = target_base >> 12; // should have low 9 bits = 0 since 0x200000 >> 12 = 0x200
        bus.write64(
            l1_pt + 1 * 8,
            leaf_pte(ppn, PTE_R | PTE_W | PTE_X | PTE_A | PTE_D),
        );

        // Access at offset 0x1234 within the 2MiB page
        let result = mmu.translate(
            vaddr + 0x1234,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_base + 0x1234));
    }

    #[test]
    fn test_sv39_1g_superpage() {
        let root_pt = DRAM_BASE + 0x1_0000;
        // 1GiB superpage: map VPN[2]=0 directly to DRAM_BASE (aligned to 1GiB)
        // DRAM_BASE = 0x8000_0000 is 2GiB aligned, so PPN low 18 bits are 0
        let target_base = DRAM_BASE; // 0x8000_0000

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0000_0000;
        let ppn = target_base >> 12;
        bus.write64(
            root_pt + 0 * 8,
            leaf_pte(ppn, PTE_R | PTE_W | PTE_X | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr + 0xABCD,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_base + 0xABCD));
    }

    // ======================== Sv48 ========================

    #[test]
    fn test_sv48_4k_page() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l2_pt = DRAM_BASE + 0x2_0000;
        let l1_pt = DRAM_BASE + 0x3_0000;
        let l0_pt = DRAM_BASE + 0x4_0000;
        let target_phys = DRAM_BASE + 0x5_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv48(root_pt);

        // vaddr = 0x0000_0000_0040_1000
        // VPN[3]=0, VPN[2]=0, VPN[1]=2, VPN[0]=1
        let vaddr: u64 = 0x0040_1000;

        bus.write64(root_pt + 0 * 8, pointer_pte(l2_pt));
        bus.write64(l2_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 1 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_X | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr + 0x42,
            AccessType::Execute,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys + 0x42));
    }

    // ======================== Sv57 ========================

    #[test]
    fn test_sv57_4k_page() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l3_pt = DRAM_BASE + 0x2_0000;
        let l2_pt = DRAM_BASE + 0x3_0000;
        let l1_pt = DRAM_BASE + 0x4_0000;
        let l0_pt = DRAM_BASE + 0x5_0000;
        let target_phys = DRAM_BASE + 0x6_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv57(root_pt);

        let vaddr: u64 = 0x0020_0000; // VPN[4]=0, VPN[3]=0, VPN[2]=0, VPN[1]=1, VPN[0]=0

        bus.write64(root_pt + 0 * 8, pointer_pte(l3_pt));
        bus.write64(l3_pt + 0 * 8, pointer_pte(l2_pt));
        bus.write64(l2_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 1 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr + 0x10,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys + 0x10));
    }

    // ======================== Permission checks ========================

    #[test]
    fn test_read_permission_denied() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // Execute-only page (no R, no W)
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_X | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13)); // Load page fault
    }

    #[test]
    fn test_write_permission_denied() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // Read-only page
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Write,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(15)); // Store page fault
    }

    #[test]
    fn test_execute_permission_denied() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // Read+Write, no execute
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Execute,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(12)); // Instruction page fault
    }

    #[test]
    fn test_user_page_from_supervisor_denied() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // User page (PTE_U set)
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_U | PTE_A | PTE_D),
        );

        // Supervisor without SUM should fault
        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13)); // Load page fault
    }

    #[test]
    fn test_user_page_from_supervisor_with_sum() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        // Set SUM bit (bit 18) in MSTATUS
        csrs.write_raw(MSTATUS, csrs.read(MSTATUS) | (1 << 18));

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_U | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys));
    }

    #[test]
    fn test_mxr_allows_read_on_execute_only() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        // Set MXR bit (bit 19) in MSTATUS
        csrs.write_raw(MSTATUS, csrs.read(MSTATUS) | (1 << 19));

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // Execute-only page
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_X | PTE_A | PTE_D),
        );

        // With MXR, read should succeed on X-only pages
        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys));
    }

    // ======================== Invalid PTEs ========================

    #[test]
    fn test_invalid_pte_causes_page_fault() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        // Root entry 0 is not valid (all zeros)
        let result = mmu.translate(
            0x1000,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13));
    }

    #[test]
    fn test_misaligned_superpage_faults() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0020_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        // 2MiB superpage with misaligned PPN (low 9 bits of PPN != 0)
        let bad_ppn = (DRAM_BASE >> 12) | 0x1; // low bit set → misaligned
        bus.write64(
            l1_pt + 1 * 8,
            leaf_pte(bad_ppn, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13)); // page fault due to misalignment
    }

    // ======================== Svpbmt ========================

    #[test]
    fn test_svpbmt_reserved_encoding_faults() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // Leaf with PBMT=3 (reserved) → should fault
        let pte = leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D) | PTE_PBMT_RSVD;
        bus.write64(l0_pt + 0 * 8, pte);

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13));
    }

    // ======================== Svnapot ========================

    #[test]
    fn test_svnapot_64k_page() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        // 64KiB NAPOT: ppn[3:0] = 0b0111, effective page is 16×4KiB
        // Target base physical = DRAM_BASE + 0x10_0000 (must be 64KiB aligned)
        let target_base = DRAM_BASE + 0x10_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));

        // NAPOT PTE: ppn[3:0]=0111, N bit set
        // Must fill all 16 entries in the NAPOT group since HW walks to exact VPN[0]
        let base_ppn = target_base >> 12;
        let napot_ppn = (base_ppn & !0xF) | 0x7;
        let pte = leaf_pte(napot_ppn, PTE_R | PTE_W | PTE_A | PTE_D) | PTE_N;
        for i in 0..16 {
            bus.write64(l0_pt + i * 8, pte);
        }

        // Access at offset 0x8000 (VPN[0]=8) within the 64KiB NAPOT range
        let result = mmu.translate(
            vaddr + 0x8000,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_base + 0x8000));

        // Also test offset 0 (VPN[0]=0)
        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_base));
    }

    #[test]
    fn test_svnapot_on_superpage_faults() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0020_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        // N bit on a level-1 (2MiB) superpage → reserved → fault
        let ppn = (DRAM_BASE + 0x20_0000) >> 12;
        let pte = leaf_pte(ppn, PTE_R | PTE_W | PTE_A | PTE_D) | PTE_N;
        bus.write64(l1_pt + 1 * 8, pte);

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13));
    }

    #[test]
    fn test_svnapot_invalid_pattern_faults() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));

        // NAPOT with ppn[3:0]=0b0011 (not 0b0111) → reserved → fault
        let base_ppn = (DRAM_BASE + 0x10_0000) >> 12;
        let bad_napot_ppn = (base_ppn & !0xF) | 0x3;
        let pte = leaf_pte(bad_napot_ppn, PTE_R | PTE_W | PTE_A | PTE_D) | PTE_N;
        bus.write64(l0_pt + 0 * 8, pte);

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13));
    }

    // ======================== TLB ========================

    #[test]
    fn test_tlb_caches_translation() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        // First access: TLB miss
        let _ = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(mmu.tlb_misses, 1);
        assert_eq!(mmu.tlb_hits, 0);

        // Second access: TLB hit
        let result = mmu.translate(
            vaddr + 0x10,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys + 0x10));
        assert_eq!(mmu.tlb_hits, 1);
    }

    #[test]
    fn test_tlb_flush_invalidates() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        // Populate TLB
        let _ = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(mmu.tlb_misses, 1);

        // Flush TLB
        mmu.flush_tlb();

        // Next access should miss again
        let _ = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(mmu.tlb_misses, 2);
    }

    #[test]
    fn test_tlb_flush_vaddr() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        // Populate TLB
        let _ = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );

        // Flush only that vaddr
        mmu.flush_tlb_vaddr(vaddr);

        // Should miss again
        let _ = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(mmu.tlb_misses, 2);
    }

    // ======================== Bare mode / M-mode ========================

    #[test]
    fn test_bare_mode_passthrough() {
        let (mut bus, mut csrs, mut mmu) = setup_sv39(DRAM_BASE + 0x1_0000);
        // Override SATP to bare mode (mode=0)
        csrs.write_raw(SATP, 0);

        let addr = DRAM_BASE + 0x100;
        let result = mmu.translate(
            addr,
            AccessType::Read,
            PrivilegeMode::Supervisor,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(addr));
    }

    #[test]
    fn test_machine_mode_no_translation() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);
        // Don't set up any page tables — M-mode bypasses translation

        let addr = DRAM_BASE + 0x200;
        let result = mmu.translate(
            addr,
            AccessType::Read,
            PrivilegeMode::Machine,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(addr));
    }

    // ======================== User mode ========================

    #[test]
    fn test_user_mode_access_to_user_page() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_U | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::User,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Ok(target_phys));
    }

    #[test]
    fn test_user_mode_denied_supervisor_page() {
        let root_pt = DRAM_BASE + 0x1_0000;
        let l1_pt = DRAM_BASE + 0x2_0000;
        let l0_pt = DRAM_BASE + 0x3_0000;
        let target_phys = DRAM_BASE + 0x4_0000;

        let (mut bus, mut csrs, mut mmu) = setup_sv39(root_pt);

        let vaddr: u64 = 0x0040_0000;
        bus.write64(root_pt + 0 * 8, pointer_pte(l1_pt));
        bus.write64(l1_pt + 2 * 8, pointer_pte(l0_pt));
        // No U bit → supervisor page
        bus.write64(
            l0_pt + 0 * 8,
            leaf_pte(target_phys >> 12, PTE_R | PTE_W | PTE_A | PTE_D),
        );

        let result = mmu.translate(
            vaddr,
            AccessType::Read,
            PrivilegeMode::User,
            &csrs,
            &mut bus,
        );
        assert_eq!(result, Err(13));
    }
}
