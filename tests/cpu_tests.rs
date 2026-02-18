use microvm::cpu::Cpu;
use microvm::cpu::csr;
use microvm::cpu::decode::{Instruction, expand_compressed};
use microvm::memory::{Bus, DRAM_BASE};

/// Helper: create a CPU+Bus, load instructions at DRAM_BASE, run N steps
fn run_program(instructions: &[u32], steps: usize) -> (Cpu, Bus) {
    let mut bus = Bus::new(64 * 1024);
    let bytes: Vec<u8> = instructions.iter().flat_map(|i| i.to_le_bytes()).collect();
    bus.load_binary(&bytes, 0);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    for _ in 0..steps {
        if !cpu.step(&mut bus) {
            break;
        }
    }
    (cpu, bus)
}

/// Helper: build DRAM_BASE address in register using LUI+ADDI (handles sign extension)
/// DRAM_BASE = 0x80000000. LUI 0x80000 → sign-extends to 0xFFFFFFFF80000000 on RV64.
/// We need: lui rd, 0x80001; addi rd, rd, -4096 to get 0x80000000... nah.
/// Actually simpler: use AUIPC to get PC-relative address.
/// Or: load a small offset from DRAM_BASE where the program lives.
/// Since our program is at DRAM_BASE, we use auipc x2, 0 to get current PC,
/// then add offset to reach our data area.

// ============== RV64I Base Instructions ==============

#[test]
fn test_addi() {
    let (cpu, _) = run_program(&[0x02A00093], 1);
    assert_eq!(cpu.regs[1], 42);
}

#[test]
fn test_addi_negative() {
    let (cpu, _) = run_program(&[0xFFF00093], 1);
    assert_eq!(cpu.regs[1] as i64, -1);
}

#[test]
fn test_lui() {
    let (cpu, _) = run_program(&[0x123450B7], 1); // lui x1, 0x12345
    assert_eq!(cpu.regs[1] as u32, 0x12345000);
}

#[test]
fn test_auipc() {
    let (cpu, _) = run_program(&[0x00001097], 1); // auipc x1, 1
    assert_eq!(cpu.regs[1], DRAM_BASE.wrapping_add(0x1000));
}

#[test]
fn test_add_sub() {
    let program = [
        0x00500093, // addi x1, x0, 5
        0x00300113, // addi x2, x0, 3
        0x002081B3, // add x3, x1, x2
        0x40208233, // sub x4, x1, x2
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 8);
    assert_eq!(cpu.regs[4], 2);
}

#[test]
fn test_logical_imm() {
    let program = [
        0x0FF00093, // addi x1, x0, 0xFF
        0x0F00C113, // xori x2, x1, 0xF0
        0x0F00E193, // ori x3, x1, 0xF0
        0x0F00F213, // andi x4, x1, 0xF0
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[2], 0x0F);
    assert_eq!(cpu.regs[3], 0xFF);
    assert_eq!(cpu.regs[4], 0xF0);
}

#[test]
fn test_slt() {
    let program = [
        0xFFF00093, // addi x1, x0, -1
        0x00100113, // addi x2, x0, 1
        0x0020A1B3, // slt x3, x1, x2
        0x0020B233, // sltu x4, x1, x2
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 1);
    assert_eq!(cpu.regs[4], 0);
}

#[test]
fn test_shifts() {
    let program = [
        0x00100093, // addi x1, x0, 1
        0x01009113, // slli x2, x1, 16
    ];
    let (cpu, _) = run_program(&program, 2);
    assert_eq!(cpu.regs[2], 1 << 16);
}

#[test]
fn test_jal() {
    let program = [
        0x008000EF, // jal x1, 8
        0x00100093, // addi x1, x0, 1 (skipped)
        0x00200113, // addi x2, x0, 2
    ];
    let (cpu, _) = run_program(&program, 2);
    assert_eq!(cpu.regs[1], DRAM_BASE + 4);
    assert_eq!(cpu.regs[2], 2);
}

#[test]
fn test_branch_beq() {
    let program = [
        0x00500093, // addi x1, x0, 5
        0x00500113, // addi x2, x0, 5
        0x00208463, // beq x1, x2, +8
        0x00100193, // addi x3, x0, 1 (skipped)
        0x00200213, // addi x4, x0, 2
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 0);
    assert_eq!(cpu.regs[4], 2);
}

#[test]
fn test_branch_bne() {
    let program = [
        0x00500093, // addi x1, x0, 5
        0x00300113, // addi x2, x0, 3
        0x00209463, // bne x1, x2, +8
        0x00100193, // addi x3, x0, 1 (skipped)
        0x00200213, // addi x4, x0, 2
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 0);
    assert_eq!(cpu.regs[4], 2);
}

// Use auipc to get a valid RAM address for load/store tests
#[test]
fn test_load_store_word() {
    let program = [
        0x02A00093u32, // addi x1, x0, 42
        0x00000117,    // auipc x2, 0        (x2 = PC = DRAM_BASE+4)
        0x10112023,    // sw x1, 256(x2)     (store at x2+256)
        0x10012183,    // lw x3, 256(x2)     (load from x2+256)
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 42);
}

#[test]
fn test_load_store_byte() {
    let program = [
        0x0AB00093u32, // addi x1, x0, 0xAB
        0x00000117,    // auipc x2, 0
        0x10110023,    // sb x1, 256(x2)
        0x10010183,    // lb x3, 256(x2)
        0x10014203,    // lbu x4, 256(x2)
    ];
    let (cpu, _) = run_program(&program, 5);
    assert_eq!(cpu.regs[3] as i64, 0xABu8 as i8 as i64);
    assert_eq!(cpu.regs[4], 0xAB);
}

#[test]
fn test_load_store_doubleword() {
    let program = [
        0x12300093u32, // addi x1, x0, 0x123
        0x00000117,    // auipc x2, 0
        0x10113023,    // sd x1, 256(x2)
        0x10013183,    // ld x3, 256(x2)
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 0x123);
}

// ============== RV64M Extension ==============

#[test]
fn test_mul() {
    let program = [
        0x00700093, // addi x1, x0, 7
        0x00600113, // addi x2, x0, 6
        0x022081B3, // mul x3, x1, x2
    ];
    let (cpu, _) = run_program(&program, 3);
    assert_eq!(cpu.regs[3], 42);
}

#[test]
fn test_div() {
    let program = [
        0x02A00093u32,
        0x00700113,
        0x0220C1B3, // div x3, x1, x2
        0x0220E233, // rem x4, x1, x2
    ];
    let (cpu, _) = run_program(&program, 4);
    assert_eq!(cpu.regs[3], 6);
    assert_eq!(cpu.regs[4], 0);
}

#[test]
fn test_div_by_zero() {
    let program = [
        0x02A00093u32,
        0x00000113,
        0x0220C1B3,
    ];
    let (cpu, _) = run_program(&program, 3);
    assert_eq!(cpu.regs[3], u64::MAX);
}

#[test]
fn test_divu() {
    let program = [
        0x02A00093u32,
        0x00700113,
        0x0220D1B3,
    ];
    let (cpu, _) = run_program(&program, 3);
    assert_eq!(cpu.regs[3], 6);
}

// ============== CSR Tests ==============

#[test]
fn test_csr_misa() {
    let (cpu, _) = run_program(&[0x301020F3u32], 1);
    let misa = cpu.regs[1];
    assert_eq!(misa >> 62, 2);
    assert_ne!(misa & (1 << 8), 0);  // I
    assert_ne!(misa & (1 << 12), 0); // M
    assert_ne!(misa & (1 << 0), 0);  // A
    assert_ne!(misa & (1 << 2), 0);  // C
    assert_ne!(misa & (1 << 18), 0); // S
}

#[test]
fn test_csrrw() {
    let program = [
        0x02A00093u32, // addi x1, x0, 42
        0x34009073,    // csrw mscratch, x1
        0x340020F3,    // csrrs x1, mscratch, x0
    ];
    let (cpu, _) = run_program(&program, 3);
    assert_eq!(cpu.regs[1], 42);
}

// ============== Decode Tests ==============

#[test]
fn test_decode_r_type() {
    let inst = Instruction::decode(0x002081B3);
    assert_eq!(inst.opcode, 0x33);
    assert_eq!(inst.rd, 3);
    assert_eq!(inst.rs1, 1);
    assert_eq!(inst.rs2, 2);
    assert_eq!(inst.funct3, 0);
    assert_eq!(inst.funct7, 0);
}

#[test]
fn test_decode_i_type() {
    let inst = Instruction::decode(0x02A00093);
    assert_eq!(inst.opcode, 0x13);
    assert_eq!(inst.rd, 1);
    assert_eq!(inst.rs1, 0);
    assert_eq!(inst.imm_i, 42);
}

#[test]
fn test_decode_negative_imm() {
    let inst = Instruction::decode(0xFFF00093);
    assert_eq!(inst.imm_i, -1);
}

// ============== Compressed Instructions ==============

#[test]
fn test_compressed_expand_c_nop() {
    let expanded = expand_compressed(0x0001); // C.NOP
    assert_eq!(expanded, 0x00000013); // addi x0, x0, 0
}

#[test]
fn test_compressed_expand_c_li() {
    // C.LI x1, 5: funct3=010, imm[5]=0, rd=00001, imm[4:0]=00101, op=01
    let val = (0b010 << 13) | (0 << 12) | (1 << 7) | (5 << 2) | 0b01;
    let expanded = expand_compressed(val);
    let expected = (5 << 20) | (0 << 15) | (0 << 12) | (1 << 7) | 0x13;
    assert_eq!(expanded, expected);
}

// ============== Memory Bus Tests ==============

#[test]
fn test_bus_read_write() {
    let mut bus = Bus::new(4096);
    bus.write32(DRAM_BASE, 0xDEADBEEF);
    assert_eq!(bus.read32(DRAM_BASE), 0xDEADBEEF);
    bus.write64(DRAM_BASE + 8, 0x123456789ABCDEF0);
    assert_eq!(bus.read64(DRAM_BASE + 8), 0x123456789ABCDEF0);
    bus.write8(DRAM_BASE + 16, 0x42);
    assert_eq!(bus.read8(DRAM_BASE + 16), 0x42);
    bus.write16(DRAM_BASE + 18, 0xBEEF);
    assert_eq!(bus.read16(DRAM_BASE + 18), 0xBEEF);
}

#[test]
fn test_bus_load_binary() {
    let mut bus = Bus::new(4096);
    bus.load_binary(&[0x13, 0x00, 0x00, 0x00], 0);
    assert_eq!(bus.read32(DRAM_BASE), 0x00000013);
}

// ============== CLINT Tests ==============

#[test]
fn test_clint_msip() {
    let mut bus = Bus::new(4096);
    use microvm::memory::CLINT_BASE;
    bus.write32(CLINT_BASE, 1);
    assert!(bus.clint.software_interrupt());
    bus.write32(CLINT_BASE, 0);
    assert!(!bus.clint.software_interrupt());
}

#[test]
fn test_clint_timer() {
    let mut bus = Bus::new(4096);
    use microvm::memory::CLINT_BASE;
    // Set mtimecmp low and high to 0
    bus.write32(CLINT_BASE + 0x4000, 0);
    bus.write32(CLINT_BASE + 0x4004, 0);
    assert!(bus.clint.timer_interrupt());
    // Set mtimecmp to MAX
    bus.write32(CLINT_BASE + 0x4000, 0xFFFFFFFF);
    bus.write32(CLINT_BASE + 0x4004, 0xFFFFFFFF);
    assert!(!bus.clint.timer_interrupt());
}

// ============== PLIC Tests ==============

#[test]
fn test_plic_interrupt() {
    let mut bus = Bus::new(4096);
    use microvm::memory::PLIC_BASE;
    bus.write32(PLIC_BASE + 0x28, 1);
    bus.write32(PLIC_BASE + 0x2080, 1 << 10);
    bus.write32(PLIC_BASE + 0x201000, 0);
    assert!(!bus.plic.has_interrupt(1));
    bus.plic.set_pending(10);
    assert!(bus.plic.has_interrupt(1));
}

// ============== UART Tests ==============

#[test]
fn test_uart_tx() {
    let mut bus = Bus::new(4096);
    use microvm::memory::UART_BASE;
    bus.write8(UART_BASE, b'A');
    let lsr = bus.read8(UART_BASE + 5);
    assert_ne!(lsr & 0x20, 0);
}

#[test]
fn test_uart_rx() {
    let mut bus = Bus::new(4096);
    use microvm::memory::UART_BASE;
    bus.uart.push_byte(b'X');
    let lsr = bus.read8(UART_BASE + 5);
    assert_ne!(lsr & 0x01, 0);
    let byte = bus.read8(UART_BASE);
    assert_eq!(byte, b'X');
}

// ============== Integration Tests ==============

#[test]
fn test_fibonacci() {
    // fib(10) = 89: x1=prev, x2=curr, x3=counter
    let program = [
        0x00000093u32, // addi x1, x0, 0     (prev=0)
        0x00100113,    // addi x2, x0, 1     (curr=1)
        0x00A00193,    // addi x3, x0, 10    (counter=10)
        // Loop at offset 0x0C:
        0x00208233,    // add x4, x1, x2     (temp = prev+curr)
        0x00010093,    // addi x1, x2, 0     (prev = curr)
        0x00020113,    // addi x2, x4, 0     (curr = temp)
        0xFFF18193,    // addi x3, x3, -1    (counter--)
        // bne x3, x0, -16 (back to offset 0x0C from offset 0x1C)
        0xFE0198E3,    // bne x3, x0, -16
    ];
    let (cpu, _) = run_program(&program, 100);
    assert_eq!(cpu.regs[2], 89);
}

#[test]
fn test_x0_always_zero() {
    let (cpu, _) = run_program(&[0x02A00013u32], 1);
    assert_eq!(cpu.regs[0], 0);
}

// ============== Exception Handling ==============

#[test]
fn test_ecall() {
    let (cpu, _) = run_program(&[0x00000073u32], 1);
    assert_eq!(cpu.csrs.read(csr::MCAUSE), 11);
    assert_eq!(cpu.csrs.read(csr::MEPC), DRAM_BASE);
}

#[test]
fn test_mtvec_and_trap() {
    // Set mtvec using auipc-based address, then ecall
    let program = [
        0x00000097u32, // auipc x1, 0          (x1 = DRAM_BASE)
        0x01408093,    // addi x1, x1, 20      (x1 = DRAM_BASE + 20 = handler addr)
        0x30509073,    // csrw mtvec, x1
        0x00A00213,    // addi x4, x0, 10      (marker)
        0x00000073,    // ecall → traps to DRAM_BASE+20
        // Handler at offset 20:
        0x00B00293,    // addi x5, x0, 11      (handler marker)
    ];
    let (cpu, _) = run_program(&program, 6);
    assert_eq!(cpu.regs[4], 10);
    assert_eq!(cpu.regs[5], 11);
    assert_eq!(cpu.pc, DRAM_BASE + 24); // after handler instruction
}

// ============== RV64I 32-bit Word Operations ==============

#[test]
fn test_addiw() {
    let (cpu, _) = run_program(&[0x7FF0009Bu32], 1);
    assert_eq!(cpu.regs[1], 0x7FF);
}

#[test]
fn test_addiw_sign_extend() {
    // ADDIW with result that has bit 31 set → sign-extends to 64-bit
    let program = [
        0x7FF00093u32, // addi x1, x0, 0x7FF
        0x0010809B,    // addiw x1, x1, 1 → 0x800 (positive, fits in 32-bit)
    ];
    let (cpu, _) = run_program(&program, 2);
    assert_eq!(cpu.regs[1], 0x800);
}

// ============== DTB Generation ==============

#[test]
fn test_dtb_generation() {
    let dtb = microvm::dtb::generate_dtb(128 * 1024 * 1024, "console=ttyS0", false);
    // DTB magic number
    assert_eq!(dtb[0], 0xD0);
    assert_eq!(dtb[1], 0x0D);
    assert_eq!(dtb[2], 0xFE);
    assert_eq!(dtb[3], 0xED);
    assert!(dtb.len() > 100); // Should have substantial content
}

// ============== Boot ROM ==============

#[test]
fn test_boot_rom_generation() {
    let boot = microvm::memory::rom::BootRom::generate(0x80200000, 0x87F00000);
    // Should generate valid RISC-V instructions
    assert!(boot.len() >= 24); // At least 6 instructions × 4 bytes
    // First instruction should be addi a0, zero, 0 = 0x00000513
    let first = u32::from_le_bytes([boot[0], boot[1], boot[2], boot[3]]);
    assert_eq!(first, 0x00000513);
}

// ============== SBI Call Tests ==============

#[test]
fn test_sbi_base_get_spec_version() {
    // Set up CPU in S-mode, make SBI base extension call
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    // Switch to S-mode by setting up mstatus and using mret
    // Simpler: directly set mode and registers
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.regs[17] = 0x10; // a7 = Base extension EID
    cpu.regs[16] = 0;    // a6 = get_spec_version FID
    // ECALL instruction
    let ecall = 0x00000073u32;
    bus.load_binary(&ecall.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(cpu.regs[10], 0); // SBI_SUCCESS
    assert_eq!(cpu.regs[11], 2); // spec version 0.2
}

#[test]
fn test_sbi_probe_extension() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.regs[17] = 0x10; // Base extension
    cpu.regs[16] = 3;    // probe_extension
    cpu.regs[10] = 0x54494D45; // TIME extension
    let ecall = 0x00000073u32;
    bus.load_binary(&ecall.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(cpu.regs[10], 0); // success
    assert_eq!(cpu.regs[11], 1); // available
}

#[test]
fn test_sbi_set_timer() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.regs[17] = 0; // legacy set_timer
    cpu.regs[10] = 999999999; // timer value
    // Set STIP first to verify it gets cleared
    cpu.csrs.write(csr::MIP, 1 << 5);
    let ecall = 0x00000073u32;
    bus.load_binary(&ecall.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(bus.clint.mtimecmp, 999999999);
    // STIP should be cleared
    assert_eq!(cpu.csrs.read(csr::MIP) & (1 << 5), 0);
}

#[test]
fn test_sbi_console_putchar() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.regs[17] = 1; // legacy console_putchar
    cpu.regs[10] = b'X' as u64;
    let ecall = 0x00000073u32;
    bus.load_binary(&ecall.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(cpu.regs[10], 0); // success
}

#[test]
fn test_sbi_ipi() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.regs[17] = 0x735049; // sPI extension
    cpu.regs[16] = 0; // send_ipi
    let ecall = 0x00000073u32;
    bus.load_binary(&ecall.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(cpu.regs[10], 0); // success
    // SSIP should be set
    assert_ne!(cpu.csrs.read(csr::MIP) & (1 << 1), 0);
}

#[test]
fn test_sbi_unknown_extension() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.regs[17] = 0xDEAD; // unknown extension
    let ecall = 0x00000073u32;
    bus.load_binary(&ecall.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(cpu.regs[10] as i64, -2); // SBI_ERR_NOT_SUPPORTED
}

// ============== PMP CSR Tests ==============

#[test]
fn test_pmp_csr_read_write() {
    let mut csrs = csr::CsrFile::new();
    // Write pmpcfg0
    csrs.write(0x3A0, 0x1F1F1F1F);
    assert_eq!(csrs.read(0x3A0), 0x1F1F1F1F);
    // Write pmpaddr0
    csrs.write(0x3B0, 0x8000_0000);
    assert_eq!(csrs.read(0x3B0), 0x8000_0000);
    // pmpcfg1 should be inaccessible on RV64
    csrs.write(0x3A1, 0xFFFF);
    assert_eq!(csrs.read(0x3A1), 0);
}

// ============== MSTATUS Tests ==============

#[test]
fn test_mstatus_uxl_sxl() {
    let csrs = csr::CsrFile::new();
    let mstatus = csrs.read(csr::MSTATUS);
    // UXL (bits 33:32) should be 2 (64-bit)
    assert_eq!((mstatus >> 32) & 3, 2);
    // SXL (bits 35:34) should be 2 (64-bit)
    assert_eq!((mstatus >> 34) & 3, 2);
}

#[test]
fn test_mstatus_uxl_preserved_on_write() {
    let mut csrs = csr::CsrFile::new();
    // Write MSTATUS with different UXL/SXL — should be preserved
    csrs.write(csr::MSTATUS, 0x0000_0000_0000_0008); // just MIE
    let mstatus = csrs.read(csr::MSTATUS);
    assert_eq!((mstatus >> 32) & 3, 2); // UXL still 2
    assert_eq!((mstatus >> 34) & 3, 2); // SXL still 2
    assert_eq!(mstatus & (1 << 3), 1 << 3); // MIE set
}

// ============== TIME CSR Test ==============

#[test]
fn test_time_csr_reads_mtime() {
    let mut csrs = csr::CsrFile::new();
    csrs.mtime = 12345678;
    assert_eq!(csrs.read(csr::TIME), 12345678);
}

// ============== SRET Test ==============

#[test]
fn test_sret() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    // Set up for SRET: in S-mode, return to user address
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.csrs.write(csr::SEPC, 0x80001000);
    // SPP=0 (return to U-mode), SPIE=1
    cpu.csrs.write(csr::SSTATUS, (1 << 5)); // SPIE=1
    let sret = 0x10200073u32;
    bus.load_binary(&sret.to_le_bytes(), 0);
    cpu.step(&mut bus);
    assert_eq!(cpu.pc, 0x80001000);
    assert_eq!(cpu.mode, microvm::cpu::PrivilegeMode::User);
}

// ============== Interrupt Delegation Test ==============

#[test]
fn test_interrupt_delegation_to_smode() {
    let mut bus = Bus::new(64 * 1024);
    let mut cpu = Cpu::new();
    cpu.reset(DRAM_BASE);
    // Set up: CPU in S-mode, delegate STI to S-mode
    cpu.mode = microvm::cpu::PrivilegeMode::Supervisor;
    cpu.csrs.write(csr::MIDELEG, 1 << 5); // Delegate STI
    cpu.csrs.write(csr::MIE, 1 << 5);     // Enable STI in MIE
    cpu.csrs.write(csr::STVEC, 0x80002000); // S-mode trap handler
    cpu.csrs.write(csr::MIP, 1 << 5);     // STIP pending
    // Enable SIE in MSTATUS (bit 1)
    let mstatus = cpu.csrs.read(csr::MSTATUS);
    cpu.csrs.write(csr::MSTATUS, mstatus | (1 << 1));

    // NOP at program start and at trap handler (0x80002000 = offset 0x2000)
    let nop = 0x00000013u32;
    bus.load_binary(&nop.to_le_bytes(), 0);
    bus.load_binary(&nop.to_le_bytes(), 0x2000); // trap handler

    cpu.step(&mut bus);

    // SCAUSE should be timer interrupt (delegated to S-mode)
    let scause = cpu.csrs.read(csr::SCAUSE);
    assert_eq!(scause, (1u64 << 63) | 5); // Interrupt, STI
    // SEPC should be the original PC
    assert_eq!(cpu.csrs.read(csr::SEPC), 0x80000000);
    // After executing the nop at trap handler, PC should be past it
    assert_eq!(cpu.pc, 0x80002004);
}
