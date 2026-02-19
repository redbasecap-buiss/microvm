/// GDB Remote Serial Protocol (RSP) stub server
///
/// Implements a GDB remote stub that allows debugging via `target remote :PORT`.
/// Supports: register read/write, memory read/write, single-step, continue,
/// breakpoints, and halt reason queries.
///
/// Reference: https://sourceware.org/gdb/current/onlinedocs/gdb.html/Remote-Protocol.html
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::cpu::Cpu;
use crate::memory::Bus;

/// GDB server state
pub struct GdbServer {
    listener: TcpListener,
    client: Option<TcpStream>,
    /// Software breakpoints (address → original instruction bytes)
    breakpoints: Vec<u64>,
    /// Whether we should single-step (after 's' command)
    single_step: bool,
    /// Whether the CPU is halted (waiting for GDB command)
    halted: bool,
}

impl GdbServer {
    /// Create a new GDB server listening on the given port
    pub fn new(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;
        listener.set_nonblocking(false)?;
        log::info!("GDB server listening on 127.0.0.1:{}", port);
        log::info!(
            "Connect with: riscv64-unknown-elf-gdb -ex 'target remote :{}'",
            port
        );
        Ok(Self {
            listener,
            client: None,
            breakpoints: Vec::new(),
            single_step: false,
            halted: true, // Start halted, waiting for GDB
        })
    }

    /// Wait for a GDB client to connect
    pub fn wait_for_client(&mut self) -> std::io::Result<()> {
        log::info!("Waiting for GDB client...");
        let (stream, addr) = self.listener.accept()?;
        stream.set_nonblocking(false)?;
        log::info!("GDB client connected from {}", addr);
        self.client = Some(stream);
        Ok(())
    }

    /// Main GDB interaction loop — called from VM loop when halted
    /// Returns: true = continue execution, false = detach/quit
    pub fn handle_commands(&mut self, cpu: &mut Cpu, bus: &mut Bus) -> GdbAction {
        loop {
            let packet = match self.recv_packet() {
                Some(p) => p,
                None => return GdbAction::Disconnect,
            };

            match self.process_command(&packet, cpu, bus) {
                CommandResult::Reply(resp) => {
                    self.send_packet(&resp);
                }
                CommandResult::Continue => {
                    self.single_step = false;
                    self.halted = false;
                    return GdbAction::Continue;
                }
                CommandResult::Step => {
                    self.single_step = true;
                    self.halted = false;
                    return GdbAction::Step;
                }
                CommandResult::Detach => {
                    self.send_packet("OK");
                    return GdbAction::Disconnect;
                }
            }
        }
    }

    /// Check if we should halt after an instruction (breakpoint or single-step)
    pub fn should_halt(&self, pc: u64) -> bool {
        self.single_step || self.breakpoints.contains(&pc)
    }

    /// Report a stop to GDB and enter command loop
    pub fn report_stop(&mut self, cpu: &mut Cpu, bus: &mut Bus, signal: u8) -> GdbAction {
        self.halted = true;
        let msg = format!("S{:02x}", signal);
        self.send_packet(&msg);
        self.handle_commands(cpu, bus)
    }

    // ---- Packet I/O ----

    fn recv_packet(&mut self) -> Option<String> {
        let stream = self.client.as_mut()?;
        let mut buf = [0u8; 1];

        // Wait for '$'
        loop {
            match stream.read_exact(&mut buf) {
                Ok(()) => {}
                Err(_) => return None,
            }
            if buf[0] == b'$' {
                break;
            }
            // Handle interrupt (Ctrl-C = 0x03)
            if buf[0] == 0x03 {
                return Some("\x03".to_string());
            }
        }

        // Read until '#'
        let mut data = Vec::new();
        loop {
            match stream.read_exact(&mut buf) {
                Ok(()) => {}
                Err(_) => return None,
            }
            if buf[0] == b'#' {
                break;
            }
            data.push(buf[0]);
        }

        // Read 2-char checksum (we don't verify, just ACK)
        let mut cksum = [0u8; 2];
        if stream.read_exact(&mut cksum).is_err() {
            return None;
        }

        // Send ACK
        let _ = stream.write_all(b"+");
        let _ = stream.flush();

        Some(String::from_utf8_lossy(&data).to_string())
    }

    fn send_packet(&mut self, data: &str) {
        let stream = match self.client.as_mut() {
            Some(s) => s,
            None => return,
        };

        let checksum: u8 = data.bytes().fold(0u8, |acc, b| acc.wrapping_add(b));
        let packet = format!("${}#{:02x}", data, checksum);
        let _ = stream.write_all(packet.as_bytes());
        let _ = stream.flush();

        // Wait for ACK ('+')
        let mut ack = [0u8; 1];
        let _ = stream.read_exact(&mut ack);
    }

    // ---- Command processing ----

    fn process_command(&mut self, cmd: &str, cpu: &mut Cpu, bus: &mut Bus) -> CommandResult {
        if cmd == "\x03" {
            // Ctrl-C interrupt
            return CommandResult::Reply("S02".to_string()); // SIGINT
        }

        let first = cmd.as_bytes().first().copied().unwrap_or(0);
        match first {
            b'?' => {
                // Halt reason
                CommandResult::Reply("S05".to_string()) // SIGTRAP
            }
            b'g' => {
                // Read all registers
                CommandResult::Reply(self.read_registers(cpu))
            }
            b'G' => {
                // Write all registers
                self.write_registers(cpu, &cmd[1..]);
                CommandResult::Reply("OK".to_string())
            }
            b'p' => {
                // Read single register
                let reg_num = u32::from_str_radix(&cmd[1..], 16).unwrap_or(0);
                CommandResult::Reply(self.read_single_register(cpu, reg_num))
            }
            b'P' => {
                // Write single register: Pn=value
                if let Some(eq_pos) = cmd.find('=') {
                    let reg_num = u32::from_str_radix(&cmd[1..eq_pos], 16).unwrap_or(0);
                    let val = u64::from_str_radix(&cmd[eq_pos + 1..], 16).unwrap_or(0);
                    self.write_single_register(cpu, reg_num, val);
                    CommandResult::Reply("OK".to_string())
                } else {
                    CommandResult::Reply("E01".to_string())
                }
            }
            b'm' => {
                // Read memory: maddr,length
                if let Some((addr, len)) = parse_addr_len(&cmd[1..]) {
                    CommandResult::Reply(self.read_memory(bus, addr, len))
                } else {
                    CommandResult::Reply("E01".to_string())
                }
            }
            b'M' => {
                // Write memory: Maddr,length:data
                if let Some(colon) = cmd.find(':') {
                    if let Some((addr, len)) = parse_addr_len(&cmd[1..colon]) {
                        let hex_data = &cmd[colon + 1..];
                        self.write_memory(bus, addr, len, hex_data);
                        CommandResult::Reply("OK".to_string())
                    } else {
                        CommandResult::Reply("E01".to_string())
                    }
                } else {
                    CommandResult::Reply("E01".to_string())
                }
            }
            b'c' => {
                // Continue (optionally at address)
                if cmd.len() > 1 {
                    if let Ok(addr) = u64::from_str_radix(&cmd[1..], 16) {
                        cpu.pc = addr;
                    }
                }
                CommandResult::Continue
            }
            b's' => {
                // Single step (optionally at address)
                if cmd.len() > 1 {
                    if let Ok(addr) = u64::from_str_radix(&cmd[1..], 16) {
                        cpu.pc = addr;
                    }
                }
                CommandResult::Step
            }
            b'Z' => {
                // Insert breakpoint: Z0,addr,kind
                self.insert_breakpoint(cmd)
            }
            b'z' => {
                // Remove breakpoint: z0,addr,kind
                self.remove_breakpoint(cmd)
            }
            b'D' => {
                // Detach
                CommandResult::Detach
            }
            b'k' => {
                // Kill
                CommandResult::Detach
            }
            b'q' => {
                // Query
                self.handle_query(cmd)
            }
            b'H' => {
                // Set thread — we're single-hart, always OK
                CommandResult::Reply("OK".to_string())
            }
            b'T' => {
                // Thread alive check — thread 1 is always alive
                CommandResult::Reply("OK".to_string())
            }
            b'v' => self.handle_v_command(cmd, cpu),
            _ => {
                // Unsupported command — empty response
                CommandResult::Reply(String::new())
            }
        }
    }

    fn read_registers(&self, cpu: &Cpu) -> String {
        // RISC-V GDB register order: x0-x31, pc (33 registers, 64-bit each)
        let mut hex = String::with_capacity(33 * 16);
        for i in 0..32 {
            hex.push_str(&hex_u64_le(cpu.regs[i]));
        }
        hex.push_str(&hex_u64_le(cpu.pc));
        hex
    }

    fn write_registers(&self, cpu: &mut Cpu, hex: &str) {
        for i in 0..32 {
            let start = i * 16;
            if start + 16 <= hex.len() {
                cpu.regs[i] = parse_hex_le(&hex[start..start + 16]);
            }
        }
        let pc_start = 32 * 16;
        if pc_start + 16 <= hex.len() {
            cpu.pc = parse_hex_le(&hex[pc_start..pc_start + 16]);
        }
        cpu.regs[0] = 0; // x0 always zero
    }

    fn read_single_register(&self, cpu: &Cpu, reg: u32) -> String {
        match reg {
            0..=31 => hex_u64_le(cpu.regs[reg as usize]),
            32 => hex_u64_le(cpu.pc),
            _ => "0000000000000000".to_string(),
        }
    }

    fn write_single_register(&self, cpu: &mut Cpu, reg: u32, val: u64) {
        match reg {
            0 => {} // x0 immutable
            1..=31 => cpu.regs[reg as usize] = val,
            32 => cpu.pc = val,
            _ => {}
        }
    }

    fn read_memory(&self, bus: &mut Bus, addr: u64, len: u64) -> String {
        let mut hex = String::with_capacity(len as usize * 2);
        for i in 0..len {
            let byte = bus.read8(addr + i);
            hex.push_str(&format!("{:02x}", byte));
        }
        hex
    }

    fn write_memory(&self, bus: &mut Bus, addr: u64, _len: u64, hex_data: &str) {
        let bytes: Vec<u8> = hex_data
            .as_bytes()
            .chunks(2)
            .filter_map(|chunk| {
                let s = std::str::from_utf8(chunk).ok()?;
                u8::from_str_radix(s, 16).ok()
            })
            .collect();
        for (i, &byte) in bytes.iter().enumerate() {
            bus.write8(addr + i as u64, byte);
        }
    }

    fn insert_breakpoint(&mut self, cmd: &str) -> CommandResult {
        // Z0,addr,kind — software breakpoint
        let parts: Vec<&str> = cmd[1..].split(',').collect();
        if parts.len() < 2 {
            return CommandResult::Reply("E01".to_string());
        }
        let bp_type = parts[0];
        if bp_type != "0" {
            // Only software breakpoints supported
            return CommandResult::Reply(String::new());
        }
        if let Ok(addr) = u64::from_str_radix(parts[1], 16) {
            if !self.breakpoints.contains(&addr) {
                self.breakpoints.push(addr);
            }
            CommandResult::Reply("OK".to_string())
        } else {
            CommandResult::Reply("E01".to_string())
        }
    }

    fn remove_breakpoint(&mut self, cmd: &str) -> CommandResult {
        let parts: Vec<&str> = cmd[1..].split(',').collect();
        if parts.len() < 2 {
            return CommandResult::Reply("E01".to_string());
        }
        let bp_type = parts[0];
        if bp_type != "0" {
            return CommandResult::Reply(String::new());
        }
        if let Ok(addr) = u64::from_str_radix(parts[1], 16) {
            self.breakpoints.retain(|&a| a != addr);
            CommandResult::Reply("OK".to_string())
        } else {
            CommandResult::Reply("E01".to_string())
        }
    }

    fn handle_query(&self, cmd: &str) -> CommandResult {
        if cmd.starts_with("qSupported") {
            CommandResult::Reply(
                "PacketSize=4096;swbreak+;hwbreak-;qXfer:features:read+".to_string(),
            )
        } else if cmd == "qAttached" {
            CommandResult::Reply("1".to_string())
        } else if let Some(offset_len) = cmd.strip_prefix("qXfer:features:read:target.xml:") {
            // Target description for RISC-V 64-bit
            let xml = r#"<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>riscv:rv64</architecture>
  <feature name="org.gnu.gdb.riscv.cpu">
    <reg name="zero" bitsize="64" type="int" regnum="0"/>
    <reg name="ra" bitsize="64" type="code_ptr" regnum="1"/>
    <reg name="sp" bitsize="64" type="data_ptr" regnum="2"/>
    <reg name="gp" bitsize="64" type="data_ptr" regnum="3"/>
    <reg name="tp" bitsize="64" type="data_ptr" regnum="4"/>
    <reg name="t0" bitsize="64" type="int" regnum="5"/>
    <reg name="t1" bitsize="64" type="int" regnum="6"/>
    <reg name="t2" bitsize="64" type="int" regnum="7"/>
    <reg name="s0" bitsize="64" type="data_ptr" regnum="8"/>
    <reg name="s1" bitsize="64" type="int" regnum="9"/>
    <reg name="a0" bitsize="64" type="int" regnum="10"/>
    <reg name="a1" bitsize="64" type="int" regnum="11"/>
    <reg name="a2" bitsize="64" type="int" regnum="12"/>
    <reg name="a3" bitsize="64" type="int" regnum="13"/>
    <reg name="a4" bitsize="64" type="int" regnum="14"/>
    <reg name="a5" bitsize="64" type="int" regnum="15"/>
    <reg name="a6" bitsize="64" type="int" regnum="16"/>
    <reg name="a7" bitsize="64" type="int" regnum="17"/>
    <reg name="s2" bitsize="64" type="int" regnum="18"/>
    <reg name="s3" bitsize="64" type="int" regnum="19"/>
    <reg name="s4" bitsize="64" type="int" regnum="20"/>
    <reg name="s5" bitsize="64" type="int" regnum="21"/>
    <reg name="s6" bitsize="64" type="int" regnum="22"/>
    <reg name="s7" bitsize="64" type="int" regnum="23"/>
    <reg name="s8" bitsize="64" type="int" regnum="24"/>
    <reg name="s9" bitsize="64" type="int" regnum="25"/>
    <reg name="s10" bitsize="64" type="int" regnum="26"/>
    <reg name="s11" bitsize="64" type="int" regnum="27"/>
    <reg name="t3" bitsize="64" type="int" regnum="28"/>
    <reg name="t4" bitsize="64" type="int" regnum="29"/>
    <reg name="t5" bitsize="64" type="int" regnum="30"/>
    <reg name="t6" bitsize="64" type="int" regnum="31"/>
    <reg name="pc" bitsize="64" type="code_ptr" regnum="32"/>
  </feature>
</target>"#;
            // qXfer response: l<data> (l = last, no more data)
            let parts: Vec<&str> = offset_len.split(',').collect();
            let offset = usize::from_str_radix(parts.first().unwrap_or(&"0"), 16).unwrap_or(0);
            let length = usize::from_str_radix(parts.get(1).unwrap_or(&"1000"), 16).unwrap_or(4096);
            if offset >= xml.len() {
                CommandResult::Reply("l".to_string())
            } else {
                let end = std::cmp::min(offset + length, xml.len());
                let chunk = &xml[offset..end];
                if end >= xml.len() {
                    CommandResult::Reply(format!("l{}", chunk))
                } else {
                    CommandResult::Reply(format!("m{}", chunk))
                }
            }
        } else if cmd == "qfThreadInfo" {
            CommandResult::Reply("m1".to_string()) // One thread (hart 0)
        } else if cmd == "qsThreadInfo" {
            CommandResult::Reply("l".to_string()) // No more threads
        } else if cmd == "qC" {
            CommandResult::Reply("QC1".to_string()) // Current thread = 1
        } else if let Some(hex_cmd) = cmd.strip_prefix("qRcmd,") {
            // Monitor command — decode hex and handle
            let decoded: String = hex_cmd
                .as_bytes()
                .chunks(2)
                .filter_map(|c| {
                    let s = std::str::from_utf8(c).ok()?;
                    u8::from_str_radix(s, 16).ok().map(|b| b as char)
                })
                .collect();
            self.handle_monitor(&decoded)
        } else {
            CommandResult::Reply(String::new())
        }
    }

    fn handle_v_command(&self, cmd: &str, _cpu: &mut Cpu) -> CommandResult {
        if cmd.starts_with("vMustReplyEmpty") {
            CommandResult::Reply(String::new())
        } else if cmd == "vCont?" {
            CommandResult::Reply("vCont;c;s".to_string())
        } else if cmd.starts_with("vCont;c") {
            CommandResult::Continue
        } else if cmd.starts_with("vCont;s") {
            CommandResult::Step
        } else {
            CommandResult::Reply(String::new())
        }
    }

    fn handle_monitor(&self, cmd: &str) -> CommandResult {
        let response = match cmd.trim() {
            "reset" => "Emulator reset not implemented yet\n",
            "halt" => "CPU halted\n",
            _ => "Unknown monitor command\n",
        };
        // Encode response as hex for qRcmd
        let hex: String = response.bytes().map(|b| format!("{:02x}", b)).collect();
        CommandResult::Reply(hex)
    }
}

/// What the VM should do after processing GDB commands
pub enum GdbAction {
    Continue,
    Step,
    Disconnect,
}

enum CommandResult {
    Reply(String),
    Continue,
    Step,
    Detach,
}

// ---- Helpers ----

fn hex_u64_le(val: u64) -> String {
    // GDB expects little-endian hex encoding for register values
    let bytes = val.to_le_bytes();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn parse_hex_le(hex: &str) -> u64 {
    let mut bytes = [0u8; 8];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        if i >= 8 {
            break;
        }
        if let Ok(s) = std::str::from_utf8(chunk) {
            bytes[i] = u8::from_str_radix(s, 16).unwrap_or(0);
        }
    }
    u64::from_le_bytes(bytes)
}

fn parse_addr_len(s: &str) -> Option<(u64, u64)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return None;
    }
    let addr = u64::from_str_radix(parts[0], 16).ok()?;
    let len = u64::from_str_radix(parts[1], 16).ok()?;
    Some((addr, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_u64_le() {
        assert_eq!(hex_u64_le(0), "0000000000000000");
        assert_eq!(hex_u64_le(1), "0100000000000000");
        assert_eq!(hex_u64_le(0x80200000), "0000208000000000");
    }

    #[test]
    fn test_parse_hex_le() {
        assert_eq!(parse_hex_le("0100000000000000"), 1);
        assert_eq!(parse_hex_le("0000208000000000"), 0x80200000);
    }

    #[test]
    fn test_parse_addr_len() {
        assert_eq!(parse_addr_len("80200000,100"), Some((0x80200000, 256)));
        assert_eq!(parse_addr_len("0,8"), Some((0, 8)));
        assert_eq!(parse_addr_len("invalid"), None);
    }

    #[test]
    fn test_register_read_format() {
        let cpu = Cpu::new();
        let server_breakpoints: Vec<u64> = Vec::new();
        // Verify format: 33 registers × 16 hex chars = 528 chars
        let gdb = GdbServer {
            listener: TcpListener::bind("127.0.0.1:0").unwrap(),
            client: None,
            breakpoints: server_breakpoints,
            single_step: false,
            halted: true,
        };
        let hex = gdb.read_registers(&cpu);
        assert_eq!(hex.len(), 33 * 16);
    }

    #[test]
    fn test_breakpoint_management() {
        let mut gdb = GdbServer {
            listener: TcpListener::bind("127.0.0.1:0").unwrap(),
            client: None,
            breakpoints: Vec::new(),
            single_step: false,
            halted: true,
        };

        // Insert breakpoint
        gdb.breakpoints.push(0x80200000);
        assert!(gdb.should_halt(0x80200000));
        assert!(!gdb.should_halt(0x80200004));

        // Remove breakpoint
        gdb.breakpoints.retain(|&a| a != 0x80200000);
        assert!(!gdb.should_halt(0x80200000));

        // Single-step halts everywhere
        gdb.single_step = true;
        assert!(gdb.should_halt(0x12345678));
    }
}
