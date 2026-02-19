//! RV64F (single-precision) and RV64D (double-precision) floating-point extensions.
//!
//! All f-register values are stored as u64 bits (NaN-boxed for single-precision).
//! NaN-boxing: single values stored in f-regs have upper 32 bits set to all 1s.

use super::Cpu;
use crate::memory::Bus;

/// NaN-box a 32-bit float value into 64 bits (upper 32 bits = 1s)
#[inline]
fn nan_box(val: u32) -> u64 {
    val as u64 | 0xFFFF_FFFF_0000_0000
}

/// Unbox a NaN-boxed single. If not properly NaN-boxed, return canonical NaN.
#[inline]
fn unbox_f32(val: u64) -> f32 {
    if val & 0xFFFF_FFFF_0000_0000 == 0xFFFF_FFFF_0000_0000 {
        f32::from_bits(val as u32)
    } else {
        f32::from_bits(0x7FC0_0000) // canonical NaN
    }
}

#[inline]
fn to_f64(val: u64) -> f64 {
    f64::from_bits(val)
}

// Rounding mode constants (for future use with software rounding)
// 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM, 7=dynamic (use frm CSR)

// Exception flags
const NX: u64 = 1; // Inexact
const UF: u64 = 2; // Underflow
#[allow(dead_code)]
const OF: u64 = 4; // Overflow
const DZ: u64 = 8; // Divide by zero
const NV: u64 = 16; // Invalid operation

/// Accumulate FP exception flags into FCSR
fn set_flags(cpu: &mut Cpu, flags: u64) {
    if flags != 0 {
        let fcsr = cpu.csrs.read(super::csr::FCSR);
        cpu.csrs.write(super::csr::FCSR, fcsr | (flags & 0x1F));
    }
}

/// Execute a floating-point instruction. Returns true if handled.
/// Caller must check FS != Off before calling.
pub fn execute_fp(cpu: &mut Cpu, bus: &mut Bus, raw: u32, inst_len: u64) -> bool {
    let opcode = raw & 0x7F;
    match opcode {
        0x07 => exec_fp_load(cpu, bus, raw, inst_len),
        0x27 => exec_fp_store(cpu, bus, raw, inst_len),
        0x43 => exec_fmadd(cpu, raw, inst_len),
        0x47 => exec_fmsub(cpu, raw, inst_len),
        0x4B => exec_fnmsub(cpu, raw, inst_len),
        0x4F => exec_fnmadd(cpu, raw, inst_len),
        0x53 => exec_fp_op(cpu, bus, raw, inst_len),
        _ => return false,
    }
    true
}

/// FLW / FLD
fn exec_fp_load(cpu: &mut Cpu, bus: &mut Bus, raw: u32, len: u64) {
    let rd = ((raw >> 7) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let funct3 = (raw >> 12) & 0x7;
    let imm = ((raw as i32) >> 20) as i64;
    let addr = cpu.regs[rs1].wrapping_add(imm as u64);

    let phys = match cpu
        .mmu
        .translate(addr, super::mmu::AccessType::Read, cpu.mode, &cpu.csrs, bus)
    {
        Ok(a) => a,
        Err(e) => {
            cpu.handle_exception(e, addr, bus);
            return;
        }
    };

    match funct3 {
        2 => {
            // FLW
            let val = bus.read32(phys);
            cpu.fregs[rd] = nan_box(val);
        }
        3 => {
            // FLD
            cpu.fregs[rd] = bus.read64(phys);
        }
        _ => {
            cpu.handle_exception(2, raw as u64, bus);
            return;
        }
    }
    cpu.csrs.set_fs_dirty();
    cpu.pc += len;
}

/// FSW / FSD
fn exec_fp_store(cpu: &mut Cpu, bus: &mut Bus, raw: u32, len: u64) {
    let rs2 = ((raw >> 20) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let funct3 = (raw >> 12) & 0x7;
    let imm = ((((raw >> 25) & 0x7F) as i32) << 5 | ((raw >> 7) & 0x1F) as i32) as i64;
    let imm = ((imm << 20) as i32 >> 20) as i64;
    let addr = cpu.regs[rs1].wrapping_add(imm as u64);

    let phys = match cpu.mmu.translate(
        addr,
        super::mmu::AccessType::Write,
        cpu.mode,
        &cpu.csrs,
        bus,
    ) {
        Ok(a) => a,
        Err(e) => {
            cpu.handle_exception(e, addr, bus);
            return;
        }
    };

    match funct3 {
        2 => {
            // FSW
            bus.write32(phys, cpu.fregs[rs2] as u32);
        }
        3 => {
            // FSD
            bus.write64(phys, cpu.fregs[rs2]);
        }
        _ => {
            cpu.handle_exception(2, raw as u64, bus);
            return;
        }
    }
    cpu.pc += len;
}

/// FMADD.S / FMADD.D
fn exec_fmadd(cpu: &mut Cpu, raw: u32, len: u64) {
    let rd = ((raw >> 7) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let rs2 = ((raw >> 20) & 0x1F) as usize;
    let rs3 = ((raw >> 27) & 0x1F) as usize;
    let fmt = (raw >> 25) & 0x3;
    let _rm = (raw >> 12) & 0x7;

    cpu.csrs.set_fs_dirty();
    match fmt {
        0 => {
            // FMADD.S
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let c = unbox_f32(cpu.fregs[rs3]);
            let result = a.mul_add(b, c);
            cpu.fregs[rd] = nan_box(result.to_bits());
            accumulate_f32_flags(cpu, result);
        }
        1 => {
            // FMADD.D
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let c = to_f64(cpu.fregs[rs3]);
            let result = a.mul_add(b, c);
            cpu.fregs[rd] = result.to_bits();
            accumulate_f64_flags(cpu, result);
        }
        _ => {}
    }
    cpu.pc += len;
}

/// FMSUB.S / FMSUB.D: rd = rs1*rs2 - rs3
fn exec_fmsub(cpu: &mut Cpu, raw: u32, len: u64) {
    let rd = ((raw >> 7) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let rs2 = ((raw >> 20) & 0x1F) as usize;
    let rs3 = ((raw >> 27) & 0x1F) as usize;
    let fmt = (raw >> 25) & 0x3;

    cpu.csrs.set_fs_dirty();
    match fmt {
        0 => {
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let c = unbox_f32(cpu.fregs[rs3]);
            let result = a.mul_add(b, -c);
            cpu.fregs[rd] = nan_box(result.to_bits());
            accumulate_f32_flags(cpu, result);
        }
        1 => {
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let c = to_f64(cpu.fregs[rs3]);
            let result = a.mul_add(b, -c);
            cpu.fregs[rd] = result.to_bits();
            accumulate_f64_flags(cpu, result);
        }
        _ => {}
    }
    cpu.pc += len;
}

/// FNMSUB.S / FNMSUB.D: rd = -(rs1*rs2) + rs3 = (-rs1)*rs2 + rs3
fn exec_fnmsub(cpu: &mut Cpu, raw: u32, len: u64) {
    let rd = ((raw >> 7) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let rs2 = ((raw >> 20) & 0x1F) as usize;
    let rs3 = ((raw >> 27) & 0x1F) as usize;
    let fmt = (raw >> 25) & 0x3;

    cpu.csrs.set_fs_dirty();
    match fmt {
        0 => {
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let c = unbox_f32(cpu.fregs[rs3]);
            let result = (-a).mul_add(b, c);
            cpu.fregs[rd] = nan_box(result.to_bits());
            accumulate_f32_flags(cpu, result);
        }
        1 => {
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let c = to_f64(cpu.fregs[rs3]);
            let result = (-a).mul_add(b, c);
            cpu.fregs[rd] = result.to_bits();
            accumulate_f64_flags(cpu, result);
        }
        _ => {}
    }
    cpu.pc += len;
}

/// FNMADD.S / FNMADD.D: rd = -(rs1*rs2) - rs3 = (-rs1)*rs2 - rs3
fn exec_fnmadd(cpu: &mut Cpu, raw: u32, len: u64) {
    let rd = ((raw >> 7) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let rs2 = ((raw >> 20) & 0x1F) as usize;
    let rs3 = ((raw >> 27) & 0x1F) as usize;
    let fmt = (raw >> 25) & 0x3;

    cpu.csrs.set_fs_dirty();
    match fmt {
        0 => {
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let c = unbox_f32(cpu.fregs[rs3]);
            let result = (-a).mul_add(b, -c);
            cpu.fregs[rd] = nan_box(result.to_bits());
            accumulate_f32_flags(cpu, result);
        }
        1 => {
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let c = to_f64(cpu.fregs[rs3]);
            let result = (-a).mul_add(b, -c);
            cpu.fregs[rd] = result.to_bits();
            accumulate_f64_flags(cpu, result);
        }
        _ => {}
    }
    cpu.pc += len;
}

/// OP-FP (opcode 0x53) — arithmetic, conversion, comparison, sign-injection, etc.
fn exec_fp_op(cpu: &mut Cpu, bus: &mut Bus, raw: u32, len: u64) {
    let rd = ((raw >> 7) & 0x1F) as usize;
    let rs1 = ((raw >> 15) & 0x1F) as usize;
    let rs2 = ((raw >> 20) & 0x1F) as usize;
    let funct7 = (raw >> 25) & 0x7F;
    let rm = (raw >> 12) & 0x7;
    let _frm = cpu.csrs.read(super::csr::FRM) as u32;

    match funct7 {
        // === Single-precision (F) ===
        0x00 => {
            // FADD.S
            cpu.csrs.set_fs_dirty();
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let r = a + b;
            cpu.fregs[rd] = nan_box(r.to_bits());
            accumulate_f32_flags(cpu, r);
        }
        0x04 => {
            // FSUB.S
            cpu.csrs.set_fs_dirty();
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let r = a - b;
            cpu.fregs[rd] = nan_box(r.to_bits());
            accumulate_f32_flags(cpu, r);
        }
        0x08 => {
            // FMUL.S
            cpu.csrs.set_fs_dirty();
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let r = a * b;
            cpu.fregs[rd] = nan_box(r.to_bits());
            accumulate_f32_flags(cpu, r);
        }
        0x0C => {
            // FDIV.S
            cpu.csrs.set_fs_dirty();
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            if b == 0.0 && !a.is_nan() {
                set_flags(cpu, DZ);
            }
            let r = a / b;
            cpu.fregs[rd] = nan_box(r.to_bits());
            accumulate_f32_flags(cpu, r);
        }
        0x2C => {
            // FSQRT.S
            cpu.csrs.set_fs_dirty();
            let a = unbox_f32(cpu.fregs[rs1]);
            if a < 0.0 && !a.is_nan() {
                set_flags(cpu, NV);
            }
            let r = a.sqrt();
            cpu.fregs[rd] = nan_box(r.to_bits());
            accumulate_f32_flags(cpu, r);
        }
        0x10 => {
            // FSGNJ.S / FSGNJN.S / FSGNJX.S
            cpu.csrs.set_fs_dirty();
            let a = cpu.fregs[rs1] as u32;
            let b = cpu.fregs[rs2] as u32;
            let result = match rm {
                0 => (a & 0x7FFF_FFFF) | (b & 0x8000_0000), // FSGNJ
                1 => (a & 0x7FFF_FFFF) | ((b ^ 0x8000_0000) & 0x8000_0000), // FSGNJN
                2 => (a & 0x7FFF_FFFF) | ((a ^ b) & 0x8000_0000), // FSGNJX
                _ => a,
            };
            cpu.fregs[rd] = nan_box(result);
        }
        0x14 => {
            // FMIN.S / FMAX.S
            cpu.csrs.set_fs_dirty();
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let r = if a.is_nan() && b.is_nan() {
                set_flags(cpu, NV);
                f32::from_bits(0x7FC0_0000) // canonical NaN
            } else if a.is_nan() {
                if is_snan_f32(a) {
                    set_flags(cpu, NV);
                }
                b
            } else if b.is_nan() {
                if is_snan_f32(b) {
                    set_flags(cpu, NV);
                }
                a
            } else {
                match rm {
                    0 => {
                        // FMIN
                        if a == 0.0 && b == 0.0 {
                            if a.is_sign_negative() {
                                a
                            } else {
                                b
                            }
                        } else if a < b {
                            a
                        } else {
                            b
                        }
                    }
                    1 => {
                        // FMAX
                        if a == 0.0 && b == 0.0 {
                            if a.is_sign_positive() {
                                a
                            } else {
                                b
                            }
                        } else if a > b {
                            a
                        } else {
                            b
                        }
                    }
                    _ => a,
                }
            };
            cpu.fregs[rd] = nan_box(r.to_bits());
        }
        0x50 => {
            // FLE.S / FLT.S / FEQ.S (result goes to integer rd)
            let a = unbox_f32(cpu.fregs[rs1]);
            let b = unbox_f32(cpu.fregs[rs2]);
            let result = match rm {
                0 => {
                    // FLE.S
                    if a.is_nan() || b.is_nan() {
                        set_flags(cpu, NV);
                        0u64
                    } else {
                        (a <= b) as u64
                    }
                }
                1 => {
                    // FLT.S
                    if a.is_nan() || b.is_nan() {
                        set_flags(cpu, NV);
                        0u64
                    } else {
                        (a < b) as u64
                    }
                }
                2 => {
                    // FEQ.S
                    if is_snan_f32(a) || is_snan_f32(b) {
                        set_flags(cpu, NV);
                    }
                    if a.is_nan() || b.is_nan() {
                        0u64
                    } else {
                        (a == b) as u64
                    }
                }
                _ => 0u64,
            };
            cpu.regs[rd] = result;
            cpu.pc += len;
            return;
        }
        0x60 => {
            // FCVT.W.S / FCVT.WU.S / FCVT.L.S / FCVT.LU.S
            let a = unbox_f32(cpu.fregs[rs1]);
            let result = match rs2 {
                0 => fcvt_to_i32(cpu, a as f64) as u64, // FCVT.W.S
                1 => fcvt_to_u32(cpu, a as f64) as u64, // FCVT.WU.S
                2 => fcvt_to_i64(cpu, a as f64),        // FCVT.L.S
                3 => fcvt_to_u64(cpu, a as f64),        // FCVT.LU.S
                _ => 0,
            };
            cpu.regs[rd] = result;
            cpu.pc += len;
            return;
        }
        0x68 => {
            // FCVT.S.W / FCVT.S.WU / FCVT.S.L / FCVT.S.LU
            cpu.csrs.set_fs_dirty();
            let r = match rs2 {
                0 => (cpu.regs[rs1] as i32 as f32).to_bits(), // FCVT.S.W
                1 => (cpu.regs[rs1] as u32 as f32).to_bits(), // FCVT.S.WU
                2 => (cpu.regs[rs1] as i64 as f32).to_bits(), // FCVT.S.L
                3 => (cpu.regs[rs1] as f32).to_bits(),        // FCVT.S.LU
                _ => 0,
            };
            cpu.fregs[rd] = nan_box(r);
        }
        0x70 => {
            match rm {
                0 => {
                    // FMV.X.W (move fp bits to integer reg)
                    cpu.regs[rd] = cpu.fregs[rs1] as u32 as i32 as i64 as u64;
                    cpu.pc += len;
                    return;
                }
                1 => {
                    // FCLASS.S
                    let f = unbox_f32(cpu.fregs[rs1]);
                    cpu.regs[rd] = fclass_f32(f) as u64;
                    cpu.pc += len;
                    return;
                }
                _ => {
                    cpu.handle_exception(2, raw as u64, bus);
                    return;
                }
            }
        }
        0x78 => {
            // FMV.W.X (move integer bits to fp reg)
            cpu.csrs.set_fs_dirty();
            cpu.fregs[rd] = nan_box(cpu.regs[rs1] as u32);
        }

        // === Double-precision (D) ===
        0x01 => {
            // FADD.D
            cpu.csrs.set_fs_dirty();
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let r = a + b;
            cpu.fregs[rd] = r.to_bits();
            accumulate_f64_flags(cpu, r);
        }
        0x05 => {
            // FSUB.D
            cpu.csrs.set_fs_dirty();
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let r = a - b;
            cpu.fregs[rd] = r.to_bits();
            accumulate_f64_flags(cpu, r);
        }
        0x09 => {
            // FMUL.D
            cpu.csrs.set_fs_dirty();
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let r = a * b;
            cpu.fregs[rd] = r.to_bits();
            accumulate_f64_flags(cpu, r);
        }
        0x0D => {
            // FDIV.D
            cpu.csrs.set_fs_dirty();
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            if b == 0.0 && !a.is_nan() {
                set_flags(cpu, DZ);
            }
            let r = a / b;
            cpu.fregs[rd] = r.to_bits();
            accumulate_f64_flags(cpu, r);
        }
        0x2D => {
            // FSQRT.D
            cpu.csrs.set_fs_dirty();
            let a = to_f64(cpu.fregs[rs1]);
            if a < 0.0 && !a.is_nan() {
                set_flags(cpu, NV);
            }
            let r = a.sqrt();
            cpu.fregs[rd] = r.to_bits();
            accumulate_f64_flags(cpu, r);
        }
        0x11 => {
            // FSGNJ.D / FSGNJN.D / FSGNJX.D
            cpu.csrs.set_fs_dirty();
            let a = cpu.fregs[rs1];
            let b = cpu.fregs[rs2];
            let result = match rm {
                0 => (a & 0x7FFF_FFFF_FFFF_FFFF) | (b & 0x8000_0000_0000_0000),
                1 => {
                    (a & 0x7FFF_FFFF_FFFF_FFFF)
                        | ((b ^ 0x8000_0000_0000_0000) & 0x8000_0000_0000_0000)
                }
                2 => (a & 0x7FFF_FFFF_FFFF_FFFF) | ((a ^ b) & 0x8000_0000_0000_0000),
                _ => a,
            };
            cpu.fregs[rd] = result;
        }
        0x15 => {
            // FMIN.D / FMAX.D
            cpu.csrs.set_fs_dirty();
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let r = if a.is_nan() && b.is_nan() {
                set_flags(cpu, NV);
                f64::from_bits(0x7FF8_0000_0000_0000) // canonical NaN
            } else if a.is_nan() {
                if is_snan_f64(a) {
                    set_flags(cpu, NV);
                }
                b
            } else if b.is_nan() {
                if is_snan_f64(b) {
                    set_flags(cpu, NV);
                }
                a
            } else {
                match rm {
                    0 => {
                        if a == 0.0 && b == 0.0 {
                            if a.is_sign_negative() {
                                a
                            } else {
                                b
                            }
                        } else if a < b {
                            a
                        } else {
                            b
                        }
                    }
                    1 => {
                        if a == 0.0 && b == 0.0 {
                            if a.is_sign_positive() {
                                a
                            } else {
                                b
                            }
                        } else if a > b {
                            a
                        } else {
                            b
                        }
                    }
                    _ => a,
                }
            };
            cpu.fregs[rd] = r.to_bits();
        }
        0x51 => {
            // FLE.D / FLT.D / FEQ.D
            let a = to_f64(cpu.fregs[rs1]);
            let b = to_f64(cpu.fregs[rs2]);
            let result = match rm {
                0 => {
                    if a.is_nan() || b.is_nan() {
                        set_flags(cpu, NV);
                        0u64
                    } else {
                        (a <= b) as u64
                    }
                }
                1 => {
                    if a.is_nan() || b.is_nan() {
                        set_flags(cpu, NV);
                        0u64
                    } else {
                        (a < b) as u64
                    }
                }
                2 => {
                    if is_snan_f64(a) || is_snan_f64(b) {
                        set_flags(cpu, NV);
                    }
                    if a.is_nan() || b.is_nan() {
                        0u64
                    } else {
                        (a == b) as u64
                    }
                }
                _ => 0u64,
            };
            cpu.regs[rd] = result;
            cpu.pc += len;
            return;
        }
        0x61 => {
            // FCVT.W.D / FCVT.WU.D / FCVT.L.D / FCVT.LU.D
            let a = to_f64(cpu.fregs[rs1]);
            let result = match rs2 {
                0 => fcvt_to_i32(cpu, a) as u64,
                1 => fcvt_to_u32(cpu, a) as u64,
                2 => fcvt_to_i64(cpu, a),
                3 => fcvt_to_u64(cpu, a),
                _ => 0,
            };
            cpu.regs[rd] = result;
            cpu.pc += len;
            return;
        }
        0x69 => {
            // FCVT.D.W / FCVT.D.WU / FCVT.D.L / FCVT.D.LU
            cpu.csrs.set_fs_dirty();
            let r = match rs2 {
                0 => (cpu.regs[rs1] as i32 as f64).to_bits(),
                1 => (cpu.regs[rs1] as u32 as f64).to_bits(),
                2 => (cpu.regs[rs1] as i64 as f64).to_bits(),
                3 => (cpu.regs[rs1] as f64).to_bits(),
                _ => 0,
            };
            cpu.fregs[rd] = r;
        }
        0x71 => {
            match rm {
                0 => {
                    // FMV.X.D
                    cpu.regs[rd] = cpu.fregs[rs1];
                    cpu.pc += len;
                    return;
                }
                1 => {
                    // FCLASS.D
                    let f = to_f64(cpu.fregs[rs1]);
                    cpu.regs[rd] = fclass_f64(f) as u64;
                    cpu.pc += len;
                    return;
                }
                _ => {
                    cpu.handle_exception(2, raw as u64, bus);
                    return;
                }
            }
        }
        0x79 => {
            // FMV.D.X
            cpu.csrs.set_fs_dirty();
            cpu.fregs[rd] = cpu.regs[rs1];
        }
        // Conversions between S and D
        0x20 => {
            // FCVT.S.D
            cpu.csrs.set_fs_dirty();
            let d = to_f64(cpu.fregs[rs1]);
            let s = d as f32;
            cpu.fregs[rd] = nan_box(s.to_bits());
            accumulate_f32_flags(cpu, s);
        }
        0x21 => {
            // FCVT.D.S
            cpu.csrs.set_fs_dirty();
            let s = unbox_f32(cpu.fregs[rs1]);
            let d = s as f64;
            cpu.fregs[rd] = d.to_bits();
        }
        _ => {
            log::warn!("Unknown FP op funct7={:#x} at PC={:#x}", funct7, cpu.pc);
            cpu.handle_exception(2, raw as u64, bus);
            return;
        }
    }
    cpu.pc += len;
}

// === Helper functions ===

fn is_snan_f32(f: f32) -> bool {
    let bits = f.to_bits();
    // sNaN: exponent all 1s, mantissa non-zero with bit 22 = 0
    f.is_nan() && (bits & 0x0040_0000) == 0
}

fn is_snan_f64(f: f64) -> bool {
    let bits = f.to_bits();
    f.is_nan() && (bits & 0x0008_0000_0000_0000) == 0
}

fn accumulate_f32_flags(cpu: &mut Cpu, r: f32) {
    let mut flags = 0u64;
    if r.is_nan() {
        // Could be from invalid op (already set by caller for specific cases)
    }
    if r.is_infinite() {
        // overflow may have been set
    }
    // Use a simple heuristic: check if result is subnormal (underflow + inexact)
    let bits = r.to_bits();
    let exp = (bits >> 23) & 0xFF;
    if exp == 0 && (bits & 0x007F_FFFF) != 0 {
        flags |= UF | NX;
    }
    if flags != 0 {
        set_flags(cpu, flags);
    }
}

fn accumulate_f64_flags(cpu: &mut Cpu, r: f64) {
    let bits = r.to_bits();
    let exp = (bits >> 52) & 0x7FF;
    if exp == 0 && (bits & 0x000F_FFFF_FFFF_FFFF) != 0 {
        set_flags(cpu, UF | NX);
    }
}

fn fclass_f32(f: f32) -> u32 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        if frac == 0 {
            if sign == 1 {
                1 << 0
            } else {
                1 << 7
            } // -inf / +inf  — wait, reversed
              // bit 0 = -inf, bit 7 = +inf
        } else if frac & 0x0040_0000 != 0 {
            1 << 9 // quiet NaN
        } else {
            1 << 8 // signaling NaN
        }
    } else if exp == 0 {
        if frac == 0 {
            if sign == 1 {
                1 << 3
            } else {
                1 << 4
            } // -0 / +0
        } else if sign == 1 {
            1 << 2 // negative subnormal
        } else {
            1 << 5 // positive subnormal
        }
    } else if sign == 1 {
        1 << 1 // negative normal
    } else {
        1 << 6 // positive normal
    }
}

fn fclass_f64(f: f64) -> u32 {
    let bits = f.to_bits();
    let sign = (bits >> 63) & 1;
    let exp = (bits >> 52) & 0x7FF;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;

    if exp == 0x7FF {
        if frac == 0 {
            if sign == 1 {
                1 << 0
            } else {
                1 << 7
            }
        } else if frac & 0x0008_0000_0000_0000 != 0 {
            1 << 9
        } else {
            1 << 8
        }
    } else if exp == 0 {
        if frac == 0 {
            if sign == 1 {
                1 << 3
            } else {
                1 << 4
            }
        } else if sign == 1 {
            1 << 2
        } else {
            1 << 5
        }
    } else if sign == 1 {
        1 << 1
    } else {
        1 << 6
    }
}

/// Convert f64 to i32, clamping out-of-range values (RISC-V saturation semantics)
fn fcvt_to_i32(cpu: &mut Cpu, f: f64) -> i64 {
    if f.is_nan() {
        set_flags(cpu, NV);
        return i32::MAX as i64;
    }
    let v = f as i64;
    if f >= i32::MAX as f64 {
        set_flags(cpu, NV);
        i32::MAX as i64
    } else if f < i32::MIN as f64 {
        set_flags(cpu, NV);
        i32::MIN as i64
    } else {
        // Sign-extend to 64 bits
        v as i32 as i64
    }
}

fn fcvt_to_u32(cpu: &mut Cpu, f: f64) -> i64 {
    if f.is_nan() {
        set_flags(cpu, NV);
        return u32::MAX as i32 as i64; // sign-extended
    }
    if f >= (u32::MAX as f64 + 1.0) {
        set_flags(cpu, NV);
        return u32::MAX as i32 as i64;
    }
    if f < 0.0 {
        if f > -1.0 {
            // Rounds to 0
            return 0;
        }
        set_flags(cpu, NV);
        return 0;
    }
    let v = f as u32;
    v as i32 as i64
}

fn fcvt_to_i64(cpu: &mut Cpu, f: f64) -> u64 {
    if f.is_nan() {
        set_flags(cpu, NV);
        return i64::MAX as u64;
    }
    if f >= i64::MAX as f64 {
        set_flags(cpu, NV);
        return i64::MAX as u64;
    }
    if f < i64::MIN as f64 {
        set_flags(cpu, NV);
        return i64::MIN as u64;
    }
    f as i64 as u64
}

fn fcvt_to_u64(cpu: &mut Cpu, f: f64) -> u64 {
    if f.is_nan() {
        set_flags(cpu, NV);
        return u64::MAX;
    }
    if f >= u64::MAX as f64 {
        set_flags(cpu, NV);
        return u64::MAX;
    }
    if f < 0.0 {
        if f > -1.0 {
            return 0;
        }
        set_flags(cpu, NV);
        return 0;
    }
    f as u64
}
