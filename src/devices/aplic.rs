/// RISC-V Advanced Platform-Level Interrupt Controller (APLIC)
///
/// Implements the direct delivery mode of AIA's APLIC specification.
/// Replaces PLIC for modern RISC-V interrupt handling.
///
/// Memory map (4 KiB per domain):
///   0x000: domaincfg     — domain configuration
///   0x004-0x03C: sourcecfg[1..=63] — per-source configuration
///   0x1BC0: mmsiaddrcfg   — MSI address config (not used in direct mode)
///   0x1BC4: mmsiaddrcfgh  — MSI address config high
///   0x1BC8: smsiaddrcfg   — S-mode MSI address config
///   0x1BCC: smsiaddrcfgh  — S-mode MSI address config high
///   0x1C00: setip[0..1]   — set interrupt pending
///   0x1CDC: setipnum      — set pending by number
///   0x1D00: in_clrip[0..1]— clear pending / read rectified input
///   0x1DDC: clripnum      — clear pending by number
///   0x1E00: setie[0..1]   — set interrupt enable
///   0x1EDC: setienum      — set enable by number
///   0x1F00: clrie[0..1]   — clear interrupt enable
///   0x1FDC: clrienum      — clear enable by number
///   0x2000: setipnum_le   — set pending (little-endian number, unused)
///   0x2004: setipnum_be   — set pending (big-endian number, unused)
///   0x4000+hart*32: IDC registers (direct delivery mode)
///     +0x00: idelivery    — interrupt delivery control
///     +0x04: iforce       — force interrupt
///     +0x08: ithreshold   — priority threshold
///     +0x18: topi         — top pending interrupt (read-only)
///     +0x1C: claimi       — claim top interrupt (read clears pending)
/// Maximum number of interrupt sources
const MAX_SOURCES: usize = 64;
/// Maximum number of harts
const MAX_HARTS: usize = 8;

/// Source mode (from sourcecfg)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SourceMode {
    Inactive = 0,
    Detached = 1,
    RisingEdge = 4,
    FallingEdge = 5,
    LevelHigh = 6,
    LevelLow = 7,
}

impl SourceMode {
    fn from_u32(v: u32) -> Self {
        match v & 7 {
            0 => Self::Inactive,
            1 => Self::Detached,
            4 => Self::RisingEdge,
            5 => Self::FallingEdge,
            6 => Self::LevelHigh,
            7 => Self::LevelLow,
            _ => Self::Inactive,
        }
    }
}

/// Per-source configuration
#[derive(Clone)]
struct SourceCfg {
    /// Source mode (bits 2:0) and delegate flag (bit 10)
    raw: u32,
}

impl SourceCfg {
    fn new() -> Self {
        Self { raw: 0 }
    }

    fn mode(&self) -> SourceMode {
        if self.raw & (1 << 10) != 0 {
            // Delegated — treat as inactive in this domain
            SourceMode::Inactive
        } else {
            SourceMode::from_u32(self.raw)
        }
    }
}

/// Per-hart Interrupt Delivery Control (IDC) for direct mode
#[derive(Clone)]
struct IdcState {
    /// Interrupt delivery enable (0=disabled, 1=enabled)
    idelivery: u32,
    /// Force interrupt (1=force external interrupt regardless of pending)
    iforce: u32,
    /// Priority threshold (0=all, N=only prio>N get through)
    ithreshold: u32,
}

impl IdcState {
    fn new() -> Self {
        Self {
            idelivery: 0,
            iforce: 0,
            ithreshold: 0,
        }
    }
}

impl Default for Aplic {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Aplic {
    /// Domain configuration (bit 8 = DM: 0=direct, 1=MSI; bit 7 = IE)
    domaincfg: u32,
    /// Per-source configuration (index 0 unused; sources are 1-based)
    sourcecfg: [SourceCfg; MAX_SOURCES],
    /// Pending bits (bitmap, bit N = source N pending)
    pending: [u64; 1],
    /// Enable bits (bitmap, bit N = source N enabled)
    enable: [u64; 1],
    /// Target registers: for direct mode, bits[17:0] = hart_index (7:0) | prio (7:0) | iprio (17:8)
    /// We store: target[src] = (hart_index << 18) | (prio << 0)
    target: [u32; MAX_SOURCES],
    /// Per-hart IDC state
    idc: [IdcState; MAX_HARTS],
    /// Number of harts
    num_harts: usize,
    /// Interrupt output signal per hart (set when topi changes)
    irq_pending: [bool; MAX_HARTS],
}

impl Aplic {
    pub fn new() -> Self {
        Self {
            domaincfg: 0,
            sourcecfg: std::array::from_fn(|_| SourceCfg::new()),
            pending: [0; 1],
            enable: [0; 1],
            target: [0; MAX_SOURCES],
            idc: std::array::from_fn(|_| IdcState::new()),
            num_harts: 1,
            irq_pending: [false; MAX_HARTS],
        }
    }

    pub fn with_harts(n: usize) -> Self {
        let mut aplic = Self::new();
        aplic.num_harts = n.min(MAX_HARTS);
        aplic
    }

    /// Set an interrupt source as pending
    pub fn set_pending(&mut self, source: u32) {
        if source == 0 || source as usize >= MAX_SOURCES {
            return;
        }
        let word = source as usize / 64;
        let bit = source as usize % 64;
        self.pending[word] |= 1u64 << bit;
        self.update_irq();
    }

    /// Clear a pending interrupt
    pub fn clear_pending(&mut self, source: u32) {
        if source == 0 || source as usize >= MAX_SOURCES {
            return;
        }
        let word = source as usize / 64;
        let bit = source as usize % 64;
        self.pending[word] &= !(1u64 << bit);
        self.update_irq();
    }

    /// Check if any interrupt is signaled for a given hart
    #[allow(dead_code)]
    pub fn has_interrupt_for_hart(&self, hart: usize) -> bool {
        if hart >= self.num_harts {
            return false;
        }
        self.irq_pending[hart]
    }

    /// Get the top pending interrupt for a hart (highest priority = lowest number)
    /// Returns (interrupt_id, priority) or (0, 0) if none
    fn topi_for_hart(&self, hart: usize) -> (u32, u32) {
        if hart >= self.num_harts {
            return (0, 0);
        }

        let ie = self.domaincfg & (1 << 8) != 0; // domain IE bit (bit 8)
        if !ie {
            return (0, 0);
        }

        let idc = &self.idc[hart];
        if idc.idelivery == 0 && idc.iforce == 0 {
            return (0, 0);
        }

        let mut best_id: u32 = 0;
        let mut best_prio: u32 = u32::MAX;

        for src in 1..MAX_SOURCES {
            // Check if pending and enabled
            let word = src / 64;
            let bit = src % 64;
            if self.pending[word] & (1u64 << bit) == 0 {
                continue;
            }
            if self.enable[word] & (1u64 << bit) == 0 {
                continue;
            }
            // Check source is active
            if self.sourcecfg[src].mode() == SourceMode::Inactive {
                continue;
            }
            // Check target hart
            let tgt = self.target[src];
            let tgt_hart = (tgt >> 18) & 0x3F;
            let tgt_prio = tgt & 0xFF;
            if tgt_hart as usize != hart {
                continue;
            }
            // Check threshold
            let prio = if tgt_prio == 0 { 256 } else { tgt_prio }; // 0 = lowest prio
            if idc.ithreshold != 0 && prio >= idc.ithreshold {
                continue;
            }
            if prio < best_prio || (prio == best_prio && (src as u32) < best_id) {
                best_id = src as u32;
                best_prio = prio;
            }
        }

        if best_id != 0 {
            (best_id, best_prio)
        } else if idc.iforce != 0 {
            // Force interrupt: return a pseudo-interrupt
            (0, 0) // iforce signals interrupt but topi returns 0
        } else {
            (0, 0)
        }
    }

    /// Update IRQ output for all harts
    fn update_irq(&mut self) {
        for h in 0..self.num_harts {
            let (id, _) = self.topi_for_hart(h);
            self.irq_pending[h] = id != 0 || self.idc[h].iforce != 0;
        }
    }

    /// Claim the top interrupt for a hart (clears pending bit, returns id<<16|prio)
    fn claim_for_hart(&mut self, hart: usize) -> u32 {
        let (id, prio) = self.topi_for_hart(hart);
        if id != 0 {
            self.clear_pending(id);
            (id << 16) | (prio & 0xFF)
        } else {
            0
        }
    }

    /// MMIO read (offset within APLIC address space)
    pub fn read(&self, offset: u64, size: u8) -> u64 {
        if size != 4 {
            return 0; // APLIC only supports 32-bit access
        }
        let off = offset as u32;
        match off {
            0x0000 => {
                // domaincfg: bit 24 = BE (always 0, little-endian), bit 8 = DM, bit 7..0 = IE etc.
                // Return with read-only bits: bits[31:24] = 0x80 (valid domain)
                0x80000000 | self.domaincfg as u64
            }
            // sourcecfg[1..63]: offset 0x004 + (src-1)*4
            0x0004..=0x00FC => {
                let src = ((off - 0x0004) / 4 + 1) as usize;
                if src < MAX_SOURCES {
                    self.sourcecfg[src].raw as u64
                } else {
                    0
                }
            }
            // setip (read pending bitmap)
            0x1C00 => self.pending[0] as u32 as u64,
            0x1C04 => (self.pending[0] >> 32) as u32 as u64,
            // in_clrip (read rectified input value = same as pending for edge-triggered)
            0x1D00 => self.pending[0] as u32 as u64,
            0x1D04 => (self.pending[0] >> 32) as u32 as u64,
            // setie (read enable bitmap)
            0x1E00 => self.enable[0] as u32 as u64,
            0x1E04 => (self.enable[0] >> 32) as u32 as u64,
            // target[1..63]: 0x3004 + (src-1)*4
            0x3004..=0x30FC => {
                let src = ((off - 0x3004) / 4 + 1) as usize;
                if src < MAX_SOURCES {
                    self.target[src] as u64
                } else {
                    0
                }
            }
            // IDC registers: base 0x4000, stride 32 bytes per hart
            0x4000..=0x40FF => {
                let hart = ((off - 0x4000) / 32) as usize;
                let reg = (off - 0x4000) % 32;
                if hart >= self.num_harts {
                    return 0;
                }
                match reg {
                    0x00 => self.idc[hart].idelivery as u64,
                    0x04 => self.idc[hart].iforce as u64,
                    0x08 => self.idc[hart].ithreshold as u64,
                    0x18 => {
                        // topi: read-only, returns (id << 16) | prio
                        let (id, prio) = self.topi_for_hart(hart);
                        ((id as u64) << 16) | (prio as u64 & 0xFF)
                    }
                    0x1C => {
                        // claimi: read returns topi value (but we can't claim on read in immutable)
                        // This is handled specially by read_mut
                        let (id, prio) = self.topi_for_hart(hart);
                        ((id as u64) << 16) | (prio as u64 & 0xFF)
                    }
                    _ => 0,
                }
            }
            _ => 0,
        }
    }

    /// MMIO read with side effects (for claimi which claims on read)
    pub fn read_mut(&mut self, offset: u64, size: u8) -> u64 {
        if size != 4 {
            return 0;
        }
        let off = offset as u32;
        // Check if this is a claimi read
        if off >= 0x4000 {
            let hart = ((off - 0x4000) / 32) as usize;
            let reg = (off - 0x4000) % 32;
            if hart < self.num_harts && reg == 0x1C {
                // claimi: claim top interrupt
                return self.claim_for_hart(hart) as u64;
            }
        }
        self.read(offset, size)
    }

    /// MMIO write
    pub fn write(&mut self, offset: u64, val: u64, size: u8) {
        if size != 4 {
            return;
        }
        let off = offset as u32;
        let val32 = val as u32;
        match off {
            0x0000 => {
                // domaincfg: only IE (bit 8) and DM (bit 2) are writable
                // In direct mode, DM=0. We support direct mode only.
                self.domaincfg = val32 & 0x100; // only keep IE bit
                self.update_irq();
            }
            0x0004..=0x00FC => {
                let src = ((off - 0x0004) / 4 + 1) as usize;
                if src < MAX_SOURCES {
                    self.sourcecfg[src].raw = val32 & 0x7FF; // bits 10:0
                    self.update_irq();
                }
            }
            // setip: OR bits into pending
            0x1C00 => {
                self.pending[0] |= (val32 as u64) & !1; // bit 0 reserved
                self.update_irq();
            }
            0x1C04 => {
                self.pending[0] |= (val32 as u64) << 32;
                self.update_irq();
            }
            // setipnum: set single source pending
            0x1CDC => {
                if val32 > 0 && (val32 as usize) < MAX_SOURCES {
                    self.set_pending(val32);
                }
            }
            // in_clrip: clear pending
            0x1D00 => {
                self.pending[0] &= !((val32 as u64) & !1);
                self.update_irq();
            }
            0x1D04 => {
                self.pending[0] &= !((val32 as u64) << 32);
                self.update_irq();
            }
            // clripnum
            0x1DDC => {
                if val32 > 0 && (val32 as usize) < MAX_SOURCES {
                    self.clear_pending(val32);
                }
            }
            // setie: OR bits into enable
            0x1E00 => {
                self.enable[0] |= (val32 as u64) & !1;
                self.update_irq();
            }
            0x1E04 => {
                self.enable[0] |= (val32 as u64) << 32;
                self.update_irq();
            }
            // setienum
            0x1EDC => {
                if val32 > 0 && (val32 as usize) < MAX_SOURCES {
                    let word = val32 as usize / 64;
                    let bit = val32 as usize % 64;
                    self.enable[word] |= 1u64 << bit;
                    self.update_irq();
                }
            }
            // clrie: clear enable bits
            0x1F00 => {
                self.enable[0] &= !((val32 as u64) & !1);
                self.update_irq();
            }
            0x1F04 => {
                self.enable[0] &= !((val32 as u64) << 32);
                self.update_irq();
            }
            // clrienum
            0x1FDC => {
                if val32 > 0 && (val32 as usize) < MAX_SOURCES {
                    let word = val32 as usize / 64;
                    let bit = val32 as usize % 64;
                    self.enable[word] &= !(1u64 << bit);
                    self.update_irq();
                }
            }
            // target registers
            0x3004..=0x30FC => {
                let src = ((off - 0x3004) / 4 + 1) as usize;
                if src < MAX_SOURCES {
                    self.target[src] = val32;
                    self.update_irq();
                }
            }
            // IDC registers
            0x4000..=0x40FF => {
                let hart = ((off - 0x4000) / 32) as usize;
                let reg = (off - 0x4000) % 32;
                if hart >= self.num_harts {
                    return;
                }
                match reg {
                    0x00 => {
                        self.idc[hart].idelivery = val32 & 1;
                        self.update_irq();
                    }
                    0x04 => {
                        self.idc[hart].iforce = val32 & 1;
                        self.update_irq();
                    }
                    0x08 => {
                        self.idc[hart].ithreshold = val32 & 0xFF;
                        self.update_irq();
                    }
                    _ => {} // topi, claimi are read-only
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::csr::{
        CsrFile, ImsicFile, IMSIC_EIDELIVERY, IMSIC_EIE0, IMSIC_EIP0, IMSIC_EITHRESHOLD, MIREG,
        MISELECT, MTOPEI, MTOPI, SIREG, SISELECT, STOPEI, STOPI,
    };

    #[test]
    fn test_aplic_new() {
        let aplic = Aplic::new();
        assert_eq!(aplic.domaincfg, 0);
        assert_eq!(aplic.num_harts, 1);
    }

    #[test]
    fn test_aplic_set_clear_pending() {
        let mut aplic = Aplic::new();
        assert_eq!(aplic.pending[0], 0);
        aplic.set_pending(10);
        assert_ne!(aplic.pending[0] & (1 << 10), 0);
        aplic.clear_pending(10);
        assert_eq!(aplic.pending[0] & (1 << 10), 0);
    }

    #[test]
    fn test_aplic_source_zero_reserved() {
        let mut aplic = Aplic::new();
        aplic.set_pending(0);
        assert_eq!(aplic.pending[0], 0);
    }

    #[test]
    fn test_aplic_domaincfg_rw() {
        let mut aplic = Aplic::new();
        // Write domaincfg with IE bit
        aplic.write(0x0000, 0x100, 4);
        // Read back: should have valid domain bit + IE
        let val = aplic.read(0x0000, 4);
        assert_eq!(val, 0x80000100);
    }

    #[test]
    fn test_aplic_sourcecfg_rw() {
        let mut aplic = Aplic::new();
        // Source 1: set to LevelHigh (6)
        aplic.write(0x0004, 6, 4);
        let val = aplic.read(0x0004, 4);
        assert_eq!(val, 6);
    }

    #[test]
    fn test_aplic_enable_rw() {
        let mut aplic = Aplic::new();
        // Enable source 5
        aplic.write(0x1EDC, 5, 4); // setienum
        let val = aplic.read(0x1E00, 4);
        assert_ne!(val & (1 << 5), 0);
        // Disable source 5
        aplic.write(0x1FDC, 5, 4); // clrienum
        let val = aplic.read(0x1E00, 4);
        assert_eq!(val & (1 << 5), 0);
    }

    #[test]
    fn test_aplic_setipnum() {
        let mut aplic = Aplic::new();
        aplic.write(0x1CDC, 3, 4); // setipnum for source 3
        let val = aplic.read(0x1C00, 4);
        assert_ne!(val & (1 << 3), 0);
    }

    #[test]
    fn test_aplic_target_rw() {
        let mut aplic = Aplic::new();
        // Set target for source 1: hart 0, priority 1
        let target_val: u32 = (0 << 18) | 1; // hart_index=0, prio=1
        aplic.write(0x3004, target_val as u64, 4);
        let readback = aplic.read(0x3004, 4);
        assert_eq!(readback, target_val as u64);
    }

    #[test]
    fn test_aplic_idc_rw() {
        let mut aplic = Aplic::new();
        // Write idelivery for hart 0
        aplic.write(0x4000, 1, 4);
        assert_eq!(aplic.read(0x4000, 4), 1);
        // Write ithreshold
        aplic.write(0x4008, 5, 4);
        assert_eq!(aplic.read(0x4008, 4), 5);
    }

    #[test]
    fn test_aplic_topi_no_interrupt() {
        let aplic = Aplic::new();
        assert_eq!(aplic.read(0x4018, 4), 0); // topi = 0 when nothing pending
    }

    #[test]
    fn test_aplic_topi_with_interrupt() {
        let mut aplic = Aplic::with_harts(1);
        // Enable domain
        aplic.write(0x0000, 0x100, 4); // IE=1
                                       // Configure source 5: LevelHigh
        aplic.write(0x0004 + 4 * 4, 6, 4); // sourcecfg[5] = LevelHigh
                                           // Set target: hart 0, prio 1
        aplic.write(0x3004 + 4 * 4, (0u32 << 18 | 1) as u64, 4);
        // Enable source 5
        aplic.write(0x1EDC, 5, 4);
        // Set pending
        aplic.set_pending(5);
        // Enable IDC delivery
        aplic.write(0x4000, 1, 4);
        // Read topi
        let topi = aplic.read(0x4018, 4);
        assert_eq!(topi >> 16, 5); // interrupt ID
        assert_eq!(topi & 0xFF, 1); // priority
    }

    #[test]
    fn test_aplic_claimi() {
        let mut aplic = Aplic::with_harts(1);
        aplic.write(0x0000, 0x100, 4); // IE
        aplic.write(0x0004 + 4 * 4, 6, 4); // source 5 LevelHigh
        aplic.write(0x3004 + 4 * 4, 1u64, 4); // target: hart 0, prio 1
        aplic.write(0x1EDC, 5, 4); // enable src 5
        aplic.set_pending(5);
        aplic.write(0x4000, 1, 4); // idelivery

        // Claim via read_mut
        let claimed = aplic.read_mut(0x401C, 4);
        assert_eq!(claimed >> 16, 5);
        // After claim, pending should be cleared
        assert_eq!(aplic.pending[0] & (1 << 5), 0);
        // topi should now be 0
        assert_eq!(aplic.read(0x4018, 4), 0);
    }

    #[test]
    fn test_aplic_threshold_filtering() {
        let mut aplic = Aplic::with_harts(1);
        aplic.write(0x0000, 0x100, 4);
        aplic.write(0x0004 + 2 * 4, 6, 4); // source 3 LevelHigh
        aplic.write(0x3004 + 2 * 4, 5u64, 4); // target: hart 0, prio 5
        aplic.write(0x1EDC, 3, 4); // enable
        aplic.set_pending(3);
        aplic.write(0x4000, 1, 4); // idelivery
                                   // Set threshold to 3 — only prio < 3 should pass
        aplic.write(0x4008, 3, 4);
        // Source has prio 5, which is >= threshold 3 → filtered out
        assert_eq!(aplic.read(0x4018, 4), 0);
    }

    #[test]
    fn test_aplic_multi_hart() {
        let mut aplic = Aplic::with_harts(2);
        aplic.write(0x0000, 0x100, 4); // IE
                                       // Source 1 → hart 0
        aplic.write(0x0004, 6, 4);
        aplic.write(0x3004, 1u64, 4); // hart 0, prio 1
                                      // Source 2 → hart 1
        aplic.write(0x0008, 6, 4);
        aplic.write(0x3008, ((1u32 << 18) | 2) as u64, 4); // hart 1, prio 2
                                                           // Enable both
        aplic.write(0x1EDC, 1, 4);
        aplic.write(0x1EDC, 2, 4);
        aplic.set_pending(1);
        aplic.set_pending(2);
        // Enable delivery for both harts
        aplic.write(0x4000, 1, 4); // hart 0 idelivery
        aplic.write(0x4020, 1, 4); // hart 1 idelivery

        assert!(aplic.has_interrupt_for_hart(0));
        assert!(aplic.has_interrupt_for_hart(1));

        // Hart 0 should see source 1
        let topi0 = aplic.read(0x4018, 4);
        assert_eq!(topi0 >> 16, 1);
        // Hart 1 should see source 2
        let topi1 = aplic.read(0x4038, 4);
        assert_eq!(topi1 >> 16, 2);
    }

    #[test]
    fn test_aplic_priority_ordering() {
        let mut aplic = Aplic::with_harts(1);
        aplic.write(0x0000, 0x100, 4);
        aplic.write(0x4000, 1, 4); // idelivery

        // Source 3: prio 5, Source 7: prio 2 (lower prio number = higher priority)
        aplic.write(0x0004 + 2 * 4, 6, 4); // src 3
        aplic.write(0x3004 + 2 * 4, 5u64, 4);
        aplic.write(0x0004 + 6 * 4, 6, 4); // src 7
        aplic.write(0x3004 + 6 * 4, 2u64, 4);

        aplic.write(0x1EDC, 3, 4);
        aplic.write(0x1EDC, 7, 4);
        aplic.set_pending(3);
        aplic.set_pending(7);

        // Source 7 has higher priority (prio 2 < prio 5)
        let topi = aplic.read(0x4018, 4);
        assert_eq!(topi >> 16, 7);
        assert_eq!(topi & 0xFF, 2);
    }

    #[test]
    fn test_aplic_iforce() {
        let mut aplic = Aplic::with_harts(1);
        aplic.write(0x0000, 0x100, 4); // IE
                                       // No sources configured/pending, but set iforce
        aplic.write(0x4004, 1, 4); // iforce for hart 0
        aplic.write(0x4000, 1, 4); // idelivery
        assert!(aplic.has_interrupt_for_hart(0));
    }

    #[test]
    fn test_aplic_domain_ie_disabled() {
        let mut aplic = Aplic::with_harts(1);
        // Don't enable domain IE
        aplic.write(0x0004, 6, 4); // source 1 active
        aplic.write(0x3004, 1u64, 4);
        aplic.write(0x1EDC, 1, 4);
        aplic.set_pending(1);
        aplic.write(0x4000, 1, 4);
        // Domain IE is off → no interrupt
        assert!(!aplic.has_interrupt_for_hart(0));
    }

    #[test]
    fn test_aplic_only_32bit_access() {
        let aplic = Aplic::new();
        assert_eq!(aplic.read(0x0000, 1), 0); // 8-bit: returns 0
        assert_eq!(aplic.read(0x0000, 2), 0); // 16-bit: returns 0
    }

    // === IMSIC tests ===

    #[test]
    fn test_imsic_file_new() {
        let f = ImsicFile::new();
        assert_eq!(f.eidelivery, 0);
        assert_eq!(f.eithreshold, 0);
        assert_eq!(f.top_pending(), 0);
    }

    #[test]
    fn test_imsic_set_pending() {
        let mut f = ImsicFile::new();
        f.eidelivery = 1;
        f.eie[0] = u64::MAX; // enable all
        f.set_pending(5);
        assert_eq!(f.top_pending(), 5);
    }

    #[test]
    fn test_imsic_claim() {
        let mut f = ImsicFile::new();
        f.eidelivery = 1;
        f.eie[0] = u64::MAX;
        f.set_pending(3);
        let id = f.claim_top();
        assert_eq!(id, 3);
        assert_eq!(f.top_pending(), 0); // cleared
    }

    #[test]
    fn test_imsic_threshold() {
        let mut f = ImsicFile::new();
        f.eidelivery = 1;
        f.eie[0] = u64::MAX;
        f.eithreshold = 3; // only IDs < 3
        f.set_pending(5);
        assert_eq!(f.top_pending(), 0); // 5 >= 3, filtered
        f.set_pending(2);
        assert_eq!(f.top_pending(), 2); // 2 < 3, passes
    }

    #[test]
    fn test_imsic_id_zero_reserved() {
        let mut f = ImsicFile::new();
        f.eidelivery = 1;
        f.eie[0] = u64::MAX;
        f.set_pending(0);
        assert_eq!(f.top_pending(), 0); // id 0 is reserved
    }

    #[test]
    fn test_imsic_indirect_rw() {
        let mut f = ImsicFile::new();
        // Write eidelivery
        f.write_indirect(IMSIC_EIDELIVERY, 1);
        assert_eq!(f.read_indirect(IMSIC_EIDELIVERY), 1);
        // Write eithreshold
        f.write_indirect(IMSIC_EITHRESHOLD, 10);
        assert_eq!(f.read_indirect(IMSIC_EITHRESHOLD), 10);
        // Write eie[0]
        f.write_indirect(IMSIC_EIE0, 0xFFFE);
        assert_eq!(f.read_indirect(IMSIC_EIE0), 0xFFFE); // bit 0 masked
                                                         // Write eip[0]
        f.write_indirect(IMSIC_EIP0, 0x1234);
        assert_eq!(f.read_indirect(IMSIC_EIP0), 0x1234); // bit 0 masked
    }

    #[test]
    fn test_imsic_disabled_no_delivery() {
        let mut f = ImsicFile::new();
        // eidelivery = 0 (disabled)
        f.eie[0] = u64::MAX;
        f.set_pending(1);
        assert_eq!(f.top_pending(), 0); // delivery disabled
    }

    #[test]
    fn test_imsic_priority_lowest_id_wins() {
        let mut f = ImsicFile::new();
        f.eidelivery = 1;
        f.eie[0] = u64::MAX;
        f.set_pending(10);
        f.set_pending(3);
        f.set_pending(7);
        assert_eq!(f.top_pending(), 3); // lowest ID = highest priority
    }

    // === AIA CSR integration tests ===

    #[test]
    fn test_aia_csr_miselect_mireg() {
        let mut csrs = CsrFile::new();
        // Set miselect to eidelivery
        csrs.write(MISELECT, IMSIC_EIDELIVERY);
        assert_eq!(csrs.read(MISELECT), IMSIC_EIDELIVERY);
        // Write mireg → writes eidelivery
        csrs.write(MIREG, 1);
        assert_eq!(csrs.imsic_m.eidelivery, 1);
        assert_eq!(csrs.read(MIREG), 1);
    }

    #[test]
    fn test_aia_csr_siselect_sireg() {
        let mut csrs = CsrFile::new();
        csrs.write(SISELECT, IMSIC_EITHRESHOLD);
        csrs.write(SIREG, 5);
        assert_eq!(csrs.imsic_s.eithreshold, 5);
        assert_eq!(csrs.read(SIREG), 5);
    }

    #[test]
    fn test_aia_csr_stopei_claim() {
        let mut csrs = CsrFile::new();
        csrs.imsic_s.eidelivery = 1;
        csrs.imsic_s.eie[0] = u64::MAX;
        csrs.imsic_s.set_pending(4);
        // Read stopi: should show interrupt 4
        let stopi = csrs.read(STOPI);
        assert_eq!(stopi >> 16, 4);
        // Read stopei: should show interrupt 4
        let stopei = csrs.read(STOPEI);
        assert_eq!(stopei >> 16, 4);
        // Write stopei: claims the interrupt
        csrs.write(STOPEI, 0);
        assert_eq!(csrs.imsic_s.top_pending(), 0);
    }

    #[test]
    fn test_aia_csr_mtopei_claim() {
        let mut csrs = CsrFile::new();
        csrs.imsic_m.eidelivery = 1;
        csrs.imsic_m.eie[0] = u64::MAX;
        csrs.imsic_m.set_pending(7);
        let mtopi = csrs.read(MTOPI);
        assert_eq!(mtopi >> 16, 7);
        csrs.write(MTOPEI, 0); // claim
        assert_eq!(csrs.imsic_m.top_pending(), 0);
    }

    #[test]
    fn test_aia_csr_eip_eie_via_indirect() {
        let mut csrs = CsrFile::new();
        // Write S-mode eie via siselect/sireg
        csrs.write(SISELECT, IMSIC_EIE0);
        csrs.write(SIREG, 0xFFFE); // enable all except bit 0
        assert_eq!(csrs.imsic_s.eie[0], 0xFFFE);
        // Write S-mode eip
        csrs.write(SISELECT, IMSIC_EIP0);
        csrs.write(SIREG, 0x0010); // pending ID 4
        assert_eq!(csrs.imsic_s.eip[0], 0x0010);
        // Enable delivery
        csrs.write(SISELECT, IMSIC_EIDELIVERY);
        csrs.write(SIREG, 1);
        assert_eq!(csrs.imsic_s.top_pending(), 4);
    }
}
