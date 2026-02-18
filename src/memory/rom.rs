/// Boot ROM — generates a minimal trampoline to jump to kernel entry
pub struct BootRom;

impl BootRom {
    /// Generate boot code that:
    /// 1. Sets a0 = hartid (0)
    /// 2. Sets a1 = DTB address
    /// 3. Jumps to kernel entry point
    ///
    /// Note: On RV64, `lui` sign-extends from 32 bits, so addresses >= 0x80000000
    /// would become 0xFFFFFFFF_80000000. We use slli+srli to zero-extend.
    pub fn generate(kernel_entry: u64, dtb_addr: u64) -> Vec<u8> {
        let mut code: Vec<u32> = Vec::new();

        // li a0, 0 (hartid)
        code.push(0x00000513); // addi a0, zero, 0

        // Load DTB address into a1 (zero-extended 64-bit)
        Self::emit_load_u64(&mut code, 11, dtb_addr); // a1 = x11

        // Load kernel entry into t0 and jump
        Self::emit_load_u64(&mut code, 5, kernel_entry); // t0 = x5
        code.push(0x00028067); // jalr zero, t0, 0 (jr t0)

        code.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Emit instructions to load a 64-bit address into register `rd`.
    /// Uses lui+addi for the low 32 bits, then slli+srli to zero-extend
    /// if bit 31 is set (which would cause lui to sign-extend on RV64).
    fn emit_load_u64(code: &mut Vec<u32>, rd: u32, addr: u64) {
        if addr <= 0x7FFFFFFF {
            // Fits in positive 32-bit range, lui+addi is fine
            let hi = ((addr.wrapping_add(0x800) >> 12) & 0xFFFFF) as u32;
            let lo = (addr & 0xFFF) as u32;
            code.push((hi << 12) | (rd << 7) | 0x37); // lui rd, hi
            code.push((lo << 20) | (rd << 15) | (0 << 12) | (rd << 7) | 0x13); // addi rd, rd, lo
        } else if addr <= 0xFFFFFFFF {
            // 32-bit address with bit 31 set — lui sign-extends, need cleanup
            let hi = ((addr.wrapping_add(0x800) >> 12) & 0xFFFFF) as u32;
            let lo = (addr & 0xFFF) as u32;
            code.push((hi << 12) | (rd << 7) | 0x37); // lui rd, hi
            code.push((lo << 20) | (rd << 15) | (0 << 12) | (rd << 7) | 0x13); // addi rd, rd, lo
            // Zero-extend: slli rd, rd, 32 then srli rd, rd, 32
            let shamt32 = 32u32;
            code.push((shamt32 << 20) | (rd << 15) | (1 << 12) | (rd << 7) | 0x13); // slli rd, rd, 32
            code.push((shamt32 << 20) | (rd << 15) | (5 << 12) | (rd << 7) | 0x13); // srli rd, rd, 32
        } else {
            // Full 64-bit: build upper 32 bits first, then shift and OR lower
            // For now, microvm only uses 32-bit physical addresses
            let hi = ((addr.wrapping_add(0x800) >> 12) & 0xFFFFF) as u32;
            let lo = (addr & 0xFFF) as u32;
            code.push((hi << 12) | (rd << 7) | 0x37);
            code.push((lo << 20) | (rd << 15) | (0 << 12) | (rd << 7) | 0x13);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_rom_generates_code() {
        let code = BootRom::generate(0x80200000, 0x87FF0000);
        assert!(!code.is_empty());
        // Should be valid little-endian RISC-V instructions
        assert_eq!(code.len() % 4, 0);
    }

    #[test]
    fn test_boot_rom_address_in_high_range() {
        // Addresses >= 0x80000000 should still produce valid code
        let code = BootRom::generate(0x80200000, 0x87F00000);
        // The code should be longer due to slli/srli fixup
        // 1 (li a0) + 4 (dtb load with fixup) + 4 (kernel load with fixup) + 1 (jr) = 10 instructions
        assert_eq!(code.len(), 10 * 4);
    }
}
