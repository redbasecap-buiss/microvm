use std::collections::VecDeque;
use std::io::{self, Write};

/// 16550 UART emulation
pub struct Uart {
    /// Receive buffer
    rx_buf: VecDeque<u8>,
    /// Line Status Register
    lsr: u8,
    /// Interrupt Enable Register
    ier: u8,
    /// Line Control Register
    lcr: u8,
    /// Modem Control Register
    mcr: u8,
    /// Divisor Latch (low/high)
    dll: u8,
    dlm: u8,
    /// FIFO Control Register
    fcr: u8,
    /// Scratch register
    scr: u8,
    /// THRE interrupt pending (cleared on IIR read when THRE is shown)
    thre_pending: bool,
}

// LSR bits
const LSR_DR: u8 = 1 << 0; // Data Ready
const LSR_THRE: u8 = 1 << 5; // Transmit Holding Register Empty
const LSR_TEMT: u8 = 1 << 6; // Transmitter Empty

// IER bits
const IER_RDA: u8 = 1 << 0; // Received Data Available
const IER_THRE: u8 = 1 << 1; // Transmitter Holding Register Empty

impl Default for Uart {
    fn default() -> Self {
        Self::new()
    }
}

impl Uart {
    pub fn new() -> Self {
        Self {
            rx_buf: VecDeque::new(),
            lsr: LSR_THRE | LSR_TEMT,
            ier: 0,
            lcr: 0,
            mcr: 0,
            dll: 0,
            dlm: 0,
            fcr: 0,
            scr: 0,
            thre_pending: true,
        }
    }

    /// Feed a byte into the receive buffer (from external source)
    pub fn push_byte(&mut self, b: u8) {
        self.rx_buf.push_back(b);
        self.lsr |= LSR_DR;
    }

    /// Check if UART has a pending interrupt
    pub fn has_interrupt(&self) -> bool {
        // RDA interrupt: data available and IER_RDA enabled
        if self.ier & IER_RDA != 0 && self.lsr & LSR_DR != 0 {
            return true;
        }
        // THRE interrupt: transmitter empty, IER_THRE enabled, and THRE condition active
        if self.ier & IER_THRE != 0 && self.thre_pending {
            return true;
        }
        false
    }

    pub fn read(&self, offset: u64) -> u64 {
        let dlab = (self.lcr >> 7) & 1;
        match offset {
            0 => {
                if dlab == 1 {
                    self.dll as u64
                } else {
                    // RBR — Receive Buffer Register
                    // We need &mut self for this, so we use interior mutability pattern
                    // For simplicity, return 0; actual read happens in read_mut
                    0
                }
            }
            1 => {
                if dlab == 1 {
                    self.dlm as u64
                } else {
                    self.ier as u64
                }
            }
            2 => self.read_iir(),
            3 => self.lcr as u64,
            4 => self.mcr as u64,
            5 => self.lsr as u64,
            6 => {
                // MSR — Modem Status Register
                // Report CTS and DSR asserted (Linux checks these)
                0x30 // CTS=1, DSR=1
            }
            7 => self.scr as u64,
            _ => 0,
        }
    }

    /// Read IIR (non-mutable version for immutable read path)
    fn read_iir(&self) -> u64 {
        let fifo_bits: u8 = if self.fcr & 1 != 0 { 0xC0 } else { 0 };
        // Priority: RLS > RDA > THRE > Modem
        if self.lsr & LSR_DR != 0 && self.ier & IER_RDA != 0 {
            (fifo_bits | 0x04) as u64 // RDA interrupt pending
        } else if self.thre_pending && self.ier & IER_THRE != 0 {
            (fifo_bits | 0x02) as u64 // THRE interrupt pending
        } else {
            (fifo_bits | 0x01) as u64 // No interrupt pending
        }
    }

    pub fn read_mut(&mut self, offset: u64) -> u64 {
        let dlab = (self.lcr >> 7) & 1;
        match offset {
            0 => {
                if dlab == 1 {
                    self.dll as u64
                } else if let Some(b) = self.rx_buf.pop_front() {
                    if self.rx_buf.is_empty() {
                        self.lsr &= !LSR_DR;
                    }
                    b as u64
                } else {
                    0
                }
            }
            2 => {
                // IIR read — reading IIR clears THRE pending (per 16550 spec)
                let val = self.read_iir();
                if val & 0x0F == 0x02 {
                    // THRE was the reported interrupt; reading IIR clears it
                    self.thre_pending = false;
                }
                val
            }
            _ => self.read(offset),
        }
    }

    /// Save UART state for snapshot
    pub fn save_state(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // rx_buf length + data
        let rx_len = self.rx_buf.len() as u32;
        out.extend_from_slice(&rx_len.to_le_bytes());
        for &b in &self.rx_buf {
            out.push(b);
        }
        // Registers
        out.push(self.lsr);
        out.push(self.ier);
        out.push(self.lcr);
        out.push(self.mcr);
        out.push(self.dll);
        out.push(self.dlm);
        out.push(self.fcr);
        out.push(self.scr);
        out.push(if self.thre_pending { 1 } else { 0 });
        out
    }

    /// Restore UART state from snapshot
    pub fn restore_state(&mut self, data: &[u8]) -> std::io::Result<()> {
        if data.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "uart",
            ));
        }
        let rx_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        self.rx_buf.clear();
        for _ in 0..rx_len {
            if pos >= data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "uart rx",
                ));
            }
            self.rx_buf.push_back(data[pos]);
            pos += 1;
        }
        if pos + 9 > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "uart regs",
            ));
        }
        self.lsr = data[pos];
        self.ier = data[pos + 1];
        self.lcr = data[pos + 2];
        self.mcr = data[pos + 3];
        self.dll = data[pos + 4];
        self.dlm = data[pos + 5];
        self.fcr = data[pos + 6];
        self.scr = data[pos + 7];
        self.thre_pending = data[pos + 8] != 0;
        Ok(())
    }

    pub fn write(&mut self, offset: u64, val: u64) {
        let val = val as u8;
        let dlab = (self.lcr >> 7) & 1;
        match offset {
            0 => {
                if dlab == 1 {
                    self.dll = val;
                } else {
                    // THR — Transmit Holding Register
                    let mut stdout = io::stdout().lock();
                    let _ = stdout.write_all(&[val]);
                    let _ = stdout.flush();
                    // Writing THR re-arms THRE interrupt (transmitter is immediately "empty" again
                    // since we flush instantly)
                    self.thre_pending = true;
                }
            }
            1 => {
                if dlab == 1 {
                    self.dlm = val;
                } else {
                    let old_ier = self.ier;
                    self.ier = val;
                    // Enabling THRE interrupt when THR is empty triggers THRE
                    if val & IER_THRE != 0 && old_ier & IER_THRE == 0 && self.lsr & LSR_THRE != 0 {
                        self.thre_pending = true;
                    }
                }
            }
            2 => self.fcr = val,
            3 => self.lcr = val,
            4 => self.mcr = val,
            7 => self.scr = val,
            _ => {}
        }
    }
}
