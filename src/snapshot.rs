//! VM snapshot/restore — save and load complete machine state.
//!
//! Binary format (little-endian):
//!   Magic:    8 bytes  "MVSN0001"
//!   CPU:      32×8B regs + 32×8B fregs + 8B pc + 8B cycle + 1B mode
//!   CSRs:     4096×8B array + 4×8B pmpcfg + 16×8B pmpaddr + 8B mtime
//!   FP:       (included in fregs above)
//!   Devices:  CLINT(24B) + PLIC(variable) + UART(small) + Syscon(1B)
//!   RAM size: 8B
//!   RAM data: compressed with zlib (length-prefixed)

use std::io::{self, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"MVSN0001";

/// Snapshot writer — collects state into a byte buffer
struct SnapshotWriter {
    buf: Vec<u8>,
}

impl SnapshotWriter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(1024 * 1024),
        }
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bytes(&mut self, data: &[u8]) {
        self.write_u64(data.len() as u64);
        self.buf.extend_from_slice(data);
    }
}

/// Snapshot reader — reads state from a byte buffer
struct SnapshotReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SnapshotReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated snapshot",
            ));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        if self.pos + 4 > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated snapshot",
            ));
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        if self.pos + 8 > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated snapshot",
            ));
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bytes(&mut self) -> io::Result<Vec<u8>> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated snapshot",
            ));
        }
        let data = self.data[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(data)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.pos + buf.len() > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated snapshot",
            ));
        }
        buf.copy_from_slice(&self.data[self.pos..self.pos + buf.len()]);
        self.pos += buf.len();
        Ok(())
    }
}

use crate::cpu::Cpu;
use crate::memory::Bus;

/// Save complete VM state to a file.
pub fn save_snapshot(path: &Path, cpu: &Cpu, bus: &mut Bus) -> io::Result<()> {
    let mut w = SnapshotWriter::new();

    // Magic
    w.buf.extend_from_slice(MAGIC);

    // CPU registers
    for &r in &cpu.regs {
        w.write_u64(r);
    }
    for &f in &cpu.fregs {
        w.write_u64(f);
    }
    w.write_u64(cpu.pc);
    w.write_u64(cpu.cycle);
    w.write_u8(cpu.mode as u8);
    w.write_u8(if cpu.wfi { 1 } else { 0 });
    // Reservation
    match cpu.reservation {
        Some(addr) => {
            w.write_u8(1);
            w.write_u64(addr);
        }
        None => {
            w.write_u8(0);
        }
    }

    // CSR file: dump all 4096 entries
    for addr in 0..4096u16 {
        w.write_u64(cpu.csrs.read_raw(addr));
    }
    // PMP state
    for &cfg in &cpu.csrs.pmpcfg {
        w.write_u64(cfg);
    }
    for &addr in &cpu.csrs.pmpaddr {
        w.write_u64(addr);
    }
    w.write_u64(cpu.csrs.mtime);

    // CLINT state
    w.write_u64(bus.clint.mtime());
    w.write_u64(bus.clint.mtimecmp());
    w.write_u32(if bus.clint.software_interrupt() { 1 } else { 0 });

    // PLIC state (priority + pending + enable + threshold + claim)
    let plic_state = bus.plic.save_state();
    w.write_bytes(&plic_state);

    // UART state
    let uart_state = bus.uart.save_state();
    w.write_bytes(&uart_state);

    // Syscon — no meaningful state to save

    // RAM
    let ram_data = bus.ram.as_slice();
    w.write_u64(ram_data.len() as u64);

    // Compress RAM with simple RLE for zero pages (most RAM is zeroed)
    let compressed = compress_ram(ram_data);
    w.write_bytes(&compressed);

    // Write to file
    let mut file = std::fs::File::create(path)?;
    file.write_all(&w.buf)?;

    log::info!(
        "Snapshot saved to {} ({:.1} MiB, RAM {:.1} MiB → {:.1} MiB compressed)",
        path.display(),
        w.buf.len() as f64 / 1024.0 / 1024.0,
        ram_data.len() as f64 / 1024.0 / 1024.0,
        compressed.len() as f64 / 1024.0 / 1024.0,
    );

    Ok(())
}

/// Load VM state from a snapshot file.
/// Returns the required RAM size so the caller can verify it matches.
pub fn load_snapshot(path: &Path, cpu: &mut Cpu, bus: &mut Bus) -> io::Result<()> {
    let data = std::fs::read(path)?;
    let mut r = SnapshotReader::new(&data);

    // Magic
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid snapshot magic: expected {:?}, got {:?}",
                MAGIC,
                std::str::from_utf8(&magic).unwrap_or("???")
            ),
        ));
    }

    // CPU registers
    for i in 0..32 {
        cpu.regs[i] = r.read_u64()?;
    }
    for i in 0..32 {
        cpu.fregs[i] = r.read_u64()?;
    }
    cpu.pc = r.read_u64()?;
    cpu.cycle = r.read_u64()?;
    cpu.mode = crate::cpu::PrivilegeMode::from_u64(r.read_u8()? as u64);
    cpu.wfi = r.read_u8()? != 0;
    // Reservation
    let has_reservation = r.read_u8()?;
    cpu.reservation = if has_reservation != 0 {
        Some(r.read_u64()?)
    } else {
        None
    };

    // CSR file
    for addr in 0..4096u16 {
        let val = r.read_u64()?;
        cpu.csrs.write_raw(addr, val);
    }
    for i in 0..4 {
        cpu.csrs.pmpcfg[i] = r.read_u64()?;
    }
    for i in 0..16 {
        cpu.csrs.pmpaddr[i] = r.read_u64()?;
    }
    cpu.csrs.mtime = r.read_u64()?;

    // Flush TLB after restoring CSRs (SATP may have changed)
    cpu.mmu.flush_tlb();

    // CLINT state
    let mtime = r.read_u64()?;
    let mtimecmp = r.read_u64()?;
    let msip = r.read_u32()?;
    bus.clint.restore_state(mtime, mtimecmp, msip != 0);

    // PLIC state
    let plic_data = r.read_bytes()?;
    bus.plic.restore_state(&plic_data)?;

    // UART state
    let uart_data = r.read_bytes()?;
    bus.uart.restore_state(&uart_data)?;

    // RAM
    let ram_size = r.read_u64()? as usize;
    if ram_size != bus.ram.as_slice().len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "RAM size mismatch: snapshot has {} MiB, VM has {} MiB",
                ram_size / 1024 / 1024,
                bus.ram.as_slice().len() / 1024 / 1024,
            ),
        ));
    }

    let compressed = r.read_bytes()?;
    let decompressed = decompress_ram(&compressed, ram_size)?;
    bus.ram.as_mut_slice().copy_from_slice(&decompressed);

    log::info!(
        "Snapshot restored from {} (PC={:#x}, mode={:?}, cycle={})",
        path.display(),
        cpu.pc,
        cpu.mode,
        cpu.cycle,
    );

    Ok(())
}

/// Simple RAM compression: page-level RLE for zero pages.
/// Format: for each 4KiB page:
///   0x00 = zero page (no data follows)
///   0x01 = non-zero page (4096 bytes follow)
fn compress_ram(data: &[u8]) -> Vec<u8> {
    const PAGE_SIZE: usize = 4096;
    let num_pages = data.len().div_ceil(PAGE_SIZE);
    // Estimate: worst case = data.len() + num_pages
    let mut out = Vec::with_capacity(data.len() / 4); // Assume mostly zero

    for i in 0..num_pages {
        let start = i * PAGE_SIZE;
        let end = std::cmp::min(start + PAGE_SIZE, data.len());
        let page = &data[start..end];

        if page.iter().all(|&b| b == 0) {
            out.push(0x00);
        } else {
            out.push(0x01);
            out.extend_from_slice(page);
        }
    }

    out
}

/// Decompress RAM from page-level RLE format.
fn decompress_ram(compressed: &[u8], expected_size: usize) -> io::Result<Vec<u8>> {
    const PAGE_SIZE: usize = 4096;
    let mut out = vec![0u8; expected_size];
    let mut pos = 0;
    let mut page_idx = 0;

    while pos < compressed.len() && page_idx * PAGE_SIZE < expected_size {
        let tag = compressed[pos];
        pos += 1;
        let page_start = page_idx * PAGE_SIZE;

        match tag {
            0x00 => {
                // Zero page — already zeroed in output
            }
            0x01 => {
                let page_end = std::cmp::min(page_start + PAGE_SIZE, expected_size);
                let page_len = page_end - page_start;
                if pos + page_len > compressed.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "truncated compressed RAM data",
                    ));
                }
                out[page_start..page_end].copy_from_slice(&compressed[pos..pos + page_len]);
                pos += page_len;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid page tag: {:#x}", tag),
                ));
            }
        }

        page_idx += 1;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_zeros() {
        let data = vec![0u8; 8192]; // 2 zero pages
        let compressed = compress_ram(&data);
        assert_eq!(compressed.len(), 2); // Just 2 tags
        let decompressed = decompress_ram(&compressed, 8192).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compress_decompress_mixed() {
        let mut data = vec![0u8; 8192]; // 2 pages
        data[0] = 0x42; // First page non-zero
        let compressed = compress_ram(&data);
        assert_eq!(compressed.len(), 1 + 4096 + 1); // tag+data + tag
        let decompressed = decompress_ram(&compressed, 8192).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compress_decompress_all_nonzero() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressed = compress_ram(&data);
        let decompressed = decompress_ram(&compressed, 4096).unwrap();
        assert_eq!(decompressed, data);
    }
}
