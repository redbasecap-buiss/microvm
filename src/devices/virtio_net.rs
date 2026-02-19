/// VirtIO MMIO Network Device (v2)
///
/// Implements a VirtIO network device with TAP backend support.
/// The device presents as a VirtIO net device (device ID 1) with
/// two virtqueues: RX (queue 0) and TX (queue 1).
///
/// When no TAP backend is attached, the device still initializes
/// and accepts packets (dropping TX, returning empty RX), allowing
/// Linux to probe and initialize the driver without errors.
///
/// Reference: VirtIO spec v1.2, sections 2, 4.2 (MMIO), 5.1 (net device).
// VirtIO MMIO register offsets
const MAGIC_VALUE: u64 = 0x000;
const VERSION: u64 = 0x004;
const DEVICE_ID: u64 = 0x008;
const VENDOR_ID: u64 = 0x00C;
const DEVICE_FEATURES: u64 = 0x010;
const DEVICE_FEATURES_SEL: u64 = 0x014;
const DRIVER_FEATURES: u64 = 0x020;
const DRIVER_FEATURES_SEL: u64 = 0x024;
const QUEUE_SEL: u64 = 0x030;
const QUEUE_NUM_MAX: u64 = 0x034;
const QUEUE_NUM: u64 = 0x038;
const QUEUE_READY: u64 = 0x044;
const QUEUE_NOTIFY: u64 = 0x050;
const INTERRUPT_STATUS: u64 = 0x060;
const INTERRUPT_ACK: u64 = 0x064;
const STATUS: u64 = 0x070;
const QUEUE_DESC_LOW: u64 = 0x080;
const QUEUE_DESC_HIGH: u64 = 0x084;
const QUEUE_AVAIL_LOW: u64 = 0x090;
const QUEUE_AVAIL_HIGH: u64 = 0x094;
const QUEUE_USED_LOW: u64 = 0x0A0;
const QUEUE_USED_HIGH: u64 = 0x0A4;
const CONFIG_GENERATION: u64 = 0x0FC;
// Config space starts at 0x100
const CONFIG_BASE: u64 = 0x100;

// VirtIO device status bits
const STATUS_DRIVER_OK: u32 = 4;

// Feature bits
const VIRTIO_NET_F_MAC: u32 = 1 << 5; // Device has given MAC address
const VIRTIO_NET_F_STATUS: u32 = 1 << 16; // Configuration status field available

// Virtqueue descriptor flags
const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

// Network link status
const VIRTIO_NET_S_LINK_UP: u16 = 1;

const QUEUE_SIZE: u32 = 256;
const NUM_QUEUES: usize = 2; // RX (0) and TX (1)

/// VirtIO queue state
struct Virtqueue {
    num: u32,
    ready: bool,
    desc_addr: u64,
    avail_addr: u64,
    used_addr: u64,
    last_avail_idx: u16,
}

impl Virtqueue {
    fn new() -> Self {
        Self {
            num: 0,
            ready: false,
            desc_addr: 0,
            avail_addr: 0,
            used_addr: 0,
            last_avail_idx: 0,
        }
    }
}

/// VirtIO MMIO Network Device
pub struct VirtioNet {
    /// Device status register
    status: u32,
    /// Device feature selection
    device_features_sel: u32,
    /// Driver feature selection
    driver_features_sel: u32,
    /// Driver features (accepted)
    driver_features: [u32; 2],
    /// Queue selection
    queue_sel: u32,
    /// Virtqueues: 0 = RX, 1 = TX
    queues: [Virtqueue; NUM_QUEUES],
    /// Interrupt status
    interrupt_status: u32,
    /// Whether an interrupt is pending
    irq_pending: bool,
    /// MAC address (6 bytes)
    mac: [u8; 6],
    /// Config generation counter
    config_gen: u32,
    /// Pending notify (which queue was notified)
    notify_queue: Option<u32>,
}

impl Default for VirtioNet {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioNet {
    pub fn new() -> Self {
        Self {
            status: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: [0; 2],
            queue_sel: 0,
            queues: [Virtqueue::new(), Virtqueue::new()],
            interrupt_status: 0,
            irq_pending: false,
            // Default MAC: 52:54:00:12:34:56 (QEMU-style locally administered)
            mac: [0x52, 0x54, 0x00, 0x12, 0x34, 0x56],
            config_gen: 0,
            notify_queue: None,
        }
    }

    /// Set the MAC address
    #[allow(dead_code)]
    pub fn set_mac(&mut self, mac: [u8; 6]) {
        self.mac = mac;
        self.config_gen = self.config_gen.wrapping_add(1);
    }

    /// Check if interrupt is pending
    pub fn has_interrupt(&self) -> bool {
        self.irq_pending
    }

    fn current_queue(&self) -> Option<&Virtqueue> {
        if (self.queue_sel as usize) < NUM_QUEUES {
            Some(&self.queues[self.queue_sel as usize])
        } else {
            None
        }
    }

    fn current_queue_mut(&mut self) -> Option<&mut Virtqueue> {
        let sel = self.queue_sel as usize;
        if sel < NUM_QUEUES {
            Some(&mut self.queues[sel])
        } else {
            None
        }
    }

    pub fn read(&self, offset: u64) -> u32 {
        match offset {
            MAGIC_VALUE => 0x74726976, // "virt"
            VERSION => 2,
            DEVICE_ID => 1,          // 1 = network device
            VENDOR_ID => 0x554D4551, // "QEMU"
            DEVICE_FEATURES => match self.device_features_sel {
                0 => VIRTIO_NET_F_MAC | VIRTIO_NET_F_STATUS,
                1 => 1, // VIRTIO_F_VERSION_1 (bit 32)
                _ => 0,
            },
            QUEUE_NUM_MAX => {
                if (self.queue_sel as usize) < NUM_QUEUES {
                    QUEUE_SIZE
                } else {
                    0
                }
            }
            QUEUE_READY => self.current_queue().map_or(0, |q| q.ready as u32),
            INTERRUPT_STATUS => self.interrupt_status,
            STATUS => self.status,
            CONFIG_GENERATION => self.config_gen,
            // Network device config space:
            // Bytes 0-5: MAC address
            // Bytes 6-7: status (u16 LE)
            c if (CONFIG_BASE..CONFIG_BASE + 8).contains(&c) => {
                let config_off = (c - CONFIG_BASE) as usize;
                self.read_config(config_off)
            }
            _ => 0,
        }
    }

    /// Read a 32-bit value from the device config space
    fn read_config(&self, offset: usize) -> u32 {
        // Config layout:
        // [0..6]  MAC address (6 bytes)
        // [6..8]  status (u16 LE)
        let mut config = [0u8; 8];
        config[..6].copy_from_slice(&self.mac);
        // Link status: always up
        let status_bytes = VIRTIO_NET_S_LINK_UP.to_le_bytes();
        config[6] = status_bytes[0];
        config[7] = status_bytes[1];

        // Return 32-bit aligned read from config
        if offset + 4 <= config.len() {
            u32::from_le_bytes([
                config[offset],
                config[offset + 1],
                config[offset + 2],
                config[offset + 3],
            ])
        } else {
            // Partial read at end of config
            let mut bytes = [0u8; 4];
            for (i, b) in bytes.iter_mut().enumerate() {
                if offset + i < config.len() {
                    *b = config[offset + i];
                }
            }
            u32::from_le_bytes(bytes)
        }
    }

    pub fn write(&mut self, offset: u64, val: u64) {
        match offset {
            DEVICE_FEATURES_SEL => self.device_features_sel = val as u32,
            DRIVER_FEATURES => {
                let sel = self.driver_features_sel as usize;
                if sel < 2 {
                    self.driver_features[sel] = val as u32;
                }
            }
            DRIVER_FEATURES_SEL => self.driver_features_sel = val as u32,
            QUEUE_SEL => self.queue_sel = val as u32,
            QUEUE_NUM => {
                if let Some(q) = self.current_queue_mut() {
                    q.num = val as u32;
                }
            }
            QUEUE_READY => {
                if let Some(q) = self.current_queue_mut() {
                    q.ready = val != 0;
                }
            }
            QUEUE_NOTIFY => {
                self.notify_queue = Some(val as u32);
            }
            INTERRUPT_ACK => {
                self.interrupt_status &= !(val as u32);
                if self.interrupt_status == 0 {
                    self.irq_pending = false;
                }
            }
            STATUS => {
                self.status = val as u32;
                if val == 0 {
                    self.reset();
                }
            }
            QUEUE_DESC_LOW => {
                if let Some(q) = self.current_queue_mut() {
                    q.desc_addr = (q.desc_addr & 0xFFFFFFFF00000000) | (val & 0xFFFFFFFF);
                }
            }
            QUEUE_DESC_HIGH => {
                if let Some(q) = self.current_queue_mut() {
                    q.desc_addr = (q.desc_addr & 0xFFFFFFFF) | ((val & 0xFFFFFFFF) << 32);
                }
            }
            QUEUE_AVAIL_LOW => {
                if let Some(q) = self.current_queue_mut() {
                    q.avail_addr = (q.avail_addr & 0xFFFFFFFF00000000) | (val & 0xFFFFFFFF);
                }
            }
            QUEUE_AVAIL_HIGH => {
                if let Some(q) = self.current_queue_mut() {
                    q.avail_addr = (q.avail_addr & 0xFFFFFFFF) | ((val & 0xFFFFFFFF) << 32);
                }
            }
            QUEUE_USED_LOW => {
                if let Some(q) = self.current_queue_mut() {
                    q.used_addr = (q.used_addr & 0xFFFFFFFF00000000) | (val & 0xFFFFFFFF);
                }
            }
            QUEUE_USED_HIGH => {
                if let Some(q) = self.current_queue_mut() {
                    q.used_addr = (q.used_addr & 0xFFFFFFFF) | ((val & 0xFFFFFFFF) << 32);
                }
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.status = 0;
        self.interrupt_status = 0;
        self.irq_pending = false;
        self.notify_queue = None;
        self.queues = [Virtqueue::new(), Virtqueue::new()];
    }

    /// Check if queue processing is needed
    pub fn needs_processing(&self) -> bool {
        self.notify_queue.is_some() && (self.status & STATUS_DRIVER_OK) != 0
    }

    /// Process pending virtqueue requests.
    /// For TX (queue 1): consume and drop packets (no backend).
    /// For RX (queue 0): nothing to deliver (no backend).
    pub fn process_queues(&mut self, ram: &mut [u8], dram_base: u64) {
        let notify = match self.notify_queue.take() {
            Some(q) => q,
            None => return,
        };

        // Process TX queue (queue 1) — consume and drop packets
        if notify == 1 {
            self.process_tx(ram, dram_base);
        }
        // RX queue (queue 0) — nothing to inject without a backend
    }

    /// Process TX queue: consume descriptor chains and drop the data.
    /// This allows the guest driver to function without stalling.
    fn process_tx(&mut self, ram: &mut [u8], dram_base: u64) {
        let q = &self.queues[1];
        if !q.ready {
            return;
        }

        let avail_addr = q.avail_addr;
        let used_addr = q.used_addr;
        let desc_addr = q.desc_addr;
        let num = q.num;
        let mut last_avail = q.last_avail_idx;

        let avail_idx = read_u16(ram, avail_addr + 2, dram_base);
        if avail_idx == last_avail {
            return;
        }

        let mut did_work = false;

        while last_avail != avail_idx {
            let ring_idx = (last_avail as u32 % num) as u64;
            let head = read_u16(ram, avail_addr + 4 + ring_idx * 2, dram_base) as u64;

            // Walk the descriptor chain to count total written bytes
            let mut total_len = 0u32;
            let mut idx = head;
            loop {
                let desc = read_descriptor(ram, dram_base, desc_addr, idx);
                // Only count device-writable descriptors in written bytes
                if desc.flags & VRING_DESC_F_WRITE != 0 {
                    total_len += desc.len;
                }
                if desc.flags & VRING_DESC_F_NEXT != 0 {
                    idx = desc.next as u64;
                } else {
                    break;
                }
            }

            // Write to used ring
            let used_idx = read_u16(ram, used_addr + 2, dram_base);
            let used_ring_idx = (used_idx as u32 % num) as u64;
            let used_elem_addr = used_addr + 4 + used_ring_idx * 8;
            write_u32(ram, used_elem_addr, dram_base, head as u32);
            write_u32(ram, used_elem_addr + 4, dram_base, total_len);
            write_u16(ram, used_addr + 2, dram_base, used_idx.wrapping_add(1));

            last_avail = last_avail.wrapping_add(1);
            did_work = true;
        }

        self.queues[1].last_avail_idx = last_avail;

        if did_work {
            self.interrupt_status |= 1;
            self.irq_pending = true;
        }
    }
}

// --- Helper functions for reading/writing guest memory ---

struct Descriptor {
    #[allow(dead_code)]
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

fn read_descriptor(ram: &[u8], dram_base: u64, desc_base: u64, idx: u64) -> Descriptor {
    let addr = desc_base + idx * 16;
    Descriptor {
        addr: read_u64(ram, addr, dram_base),
        len: read_u32(ram, addr + 8, dram_base),
        flags: read_u16(ram, addr + 12, dram_base),
        next: read_u16(ram, addr + 14, dram_base),
    }
}

fn addr_to_ram_offset(addr: u64, dram_base: u64) -> usize {
    addr.wrapping_sub(dram_base) as usize
}

fn read_u16(ram: &[u8], addr: u64, dram_base: u64) -> u16 {
    let off = addr_to_ram_offset(addr, dram_base);
    if off + 1 < ram.len() {
        u16::from_le_bytes([ram[off], ram[off + 1]])
    } else {
        0
    }
}

fn read_u32(ram: &[u8], addr: u64, dram_base: u64) -> u32 {
    let off = addr_to_ram_offset(addr, dram_base);
    if off + 3 < ram.len() {
        u32::from_le_bytes([ram[off], ram[off + 1], ram[off + 2], ram[off + 3]])
    } else {
        0
    }
}

fn read_u64(ram: &[u8], addr: u64, dram_base: u64) -> u64 {
    let off = addr_to_ram_offset(addr, dram_base);
    if off + 7 < ram.len() {
        u64::from_le_bytes([
            ram[off],
            ram[off + 1],
            ram[off + 2],
            ram[off + 3],
            ram[off + 4],
            ram[off + 5],
            ram[off + 6],
            ram[off + 7],
        ])
    } else {
        0
    }
}

fn write_u16(ram: &mut [u8], addr: u64, dram_base: u64, val: u16) {
    let off = addr_to_ram_offset(addr, dram_base);
    if off + 1 < ram.len() {
        let bytes = val.to_le_bytes();
        ram[off] = bytes[0];
        ram[off + 1] = bytes[1];
    }
}

fn write_u32(ram: &mut [u8], addr: u64, dram_base: u64, val: u32) {
    let off = addr_to_ram_offset(addr, dram_base);
    if off + 3 < ram.len() {
        let bytes = val.to_le_bytes();
        ram[off..off + 4].copy_from_slice(&bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtio_net_magic() {
        let net = VirtioNet::new();
        assert_eq!(net.read(MAGIC_VALUE), 0x74726976);
    }

    #[test]
    fn test_virtio_net_version() {
        let net = VirtioNet::new();
        assert_eq!(net.read(VERSION), 2);
    }

    #[test]
    fn test_virtio_net_device_id() {
        let net = VirtioNet::new();
        assert_eq!(net.read(DEVICE_ID), 1); // Network device
    }

    #[test]
    fn test_virtio_net_features() {
        let mut net = VirtioNet::new();
        net.write(DEVICE_FEATURES_SEL, 0);
        let feat0 = net.read(DEVICE_FEATURES);
        assert!(feat0 & VIRTIO_NET_F_MAC != 0, "MAC feature must be set");
        assert!(
            feat0 & VIRTIO_NET_F_STATUS != 0,
            "STATUS feature must be set"
        );

        net.write(DEVICE_FEATURES_SEL, 1);
        let feat1 = net.read(DEVICE_FEATURES);
        assert_eq!(feat1, 1, "VIRTIO_F_VERSION_1 must be set");
    }

    #[test]
    fn test_virtio_net_mac_config() {
        let net = VirtioNet::new();
        // Read first 4 bytes of config (MAC bytes 0-3)
        let val0 = net.read(CONFIG_BASE);
        let bytes = val0.to_le_bytes();
        assert_eq!(bytes[0], 0x52);
        assert_eq!(bytes[1], 0x54);
        assert_eq!(bytes[2], 0x00);
        assert_eq!(bytes[3], 0x12);

        // Read bytes 4-7 (MAC bytes 4-5 + status)
        let val1 = net.read(CONFIG_BASE + 4);
        let bytes = val1.to_le_bytes();
        assert_eq!(bytes[0], 0x34);
        assert_eq!(bytes[1], 0x56);
        // Status: LINK_UP = 1
        assert_eq!(
            u16::from_le_bytes([bytes[2], bytes[3]]),
            VIRTIO_NET_S_LINK_UP
        );
    }

    #[test]
    fn test_virtio_net_queue_setup() {
        let mut net = VirtioNet::new();
        // RX queue
        net.write(QUEUE_SEL, 0);
        assert_eq!(net.read(QUEUE_NUM_MAX), QUEUE_SIZE);
        net.write(QUEUE_NUM, 128);
        net.write(QUEUE_DESC_LOW, 0x80010000);
        net.write(QUEUE_DESC_HIGH, 0);
        net.write(QUEUE_AVAIL_LOW, 0x80020000);
        net.write(QUEUE_AVAIL_HIGH, 0);
        net.write(QUEUE_USED_LOW, 0x80030000);
        net.write(QUEUE_USED_HIGH, 0);
        net.write(QUEUE_READY, 1);
        assert_eq!(net.read(QUEUE_READY), 1);

        // TX queue
        net.write(QUEUE_SEL, 1);
        assert_eq!(net.read(QUEUE_NUM_MAX), QUEUE_SIZE);
        net.write(QUEUE_NUM, 128);
        net.write(QUEUE_READY, 1);
        assert_eq!(net.read(QUEUE_READY), 1);

        // Invalid queue
        net.write(QUEUE_SEL, 2);
        assert_eq!(net.read(QUEUE_NUM_MAX), 0);
    }

    #[test]
    fn test_virtio_net_status_reset() {
        let mut net = VirtioNet::new();
        net.write(STATUS, 0x0F);
        assert_eq!(net.read(STATUS), 0x0F);
        net.write(STATUS, 0); // Reset
        assert_eq!(net.read(STATUS), 0);
    }

    #[test]
    fn test_virtio_net_interrupt_ack() {
        let mut net = VirtioNet::new();
        net.interrupt_status = 1;
        net.irq_pending = true;
        assert_eq!(net.read(INTERRUPT_STATUS), 1);
        net.write(INTERRUPT_ACK, 1);
        assert_eq!(net.read(INTERRUPT_STATUS), 0);
        assert!(!net.has_interrupt());
    }

    #[test]
    fn test_virtio_net_set_mac() {
        let mut net = VirtioNet::new();
        let old_gen = net.read(CONFIG_GENERATION);
        net.set_mac([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        // Config generation should increment
        assert_eq!(net.read(CONFIG_GENERATION), old_gen + 1);
        // Verify new MAC
        let val0 = net.read(CONFIG_BASE);
        let bytes = val0.to_le_bytes();
        assert_eq!(bytes[0], 0xAA);
        assert_eq!(bytes[1], 0xBB);
        assert_eq!(bytes[2], 0xCC);
        assert_eq!(bytes[3], 0xDD);
    }

    #[test]
    fn test_virtio_net_tx_processing() {
        let mut net = VirtioNet::new();

        // Set up TX queue
        net.write(QUEUE_SEL, 1);
        net.write(QUEUE_NUM, 16);

        // We need a small RAM region to set up descriptors
        let dram_base: u64 = 0x80000000;
        let mut ram = vec![0u8; 0x10000];

        // Place descriptor table at offset 0x1000
        let desc_base: u64 = dram_base + 0x1000;
        // Place avail ring at offset 0x2000
        let avail_base: u64 = dram_base + 0x2000;
        // Place used ring at offset 0x3000
        let used_base: u64 = dram_base + 0x3000;

        net.write(QUEUE_DESC_LOW, desc_base as u64);
        net.write(QUEUE_DESC_HIGH, 0);
        net.write(QUEUE_AVAIL_LOW, avail_base as u64);
        net.write(QUEUE_AVAIL_HIGH, 0);
        net.write(QUEUE_USED_LOW, used_base as u64);
        net.write(QUEUE_USED_HIGH, 0);
        net.write(QUEUE_READY, 1);
        net.write(STATUS, STATUS_DRIVER_OK as u64);

        // Set up a single TX descriptor (virtio-net header + packet data)
        // Descriptor 0: 12-byte virtio-net header (read-only)
        let d0_off = 0x1000usize;
        let header_addr: u64 = dram_base + 0x4000;
        ram[d0_off..d0_off + 8].copy_from_slice(&header_addr.to_le_bytes());
        ram[d0_off + 8..d0_off + 12].copy_from_slice(&12u32.to_le_bytes()); // len
        ram[d0_off + 12..d0_off + 14].copy_from_slice(&VRING_DESC_F_NEXT.to_le_bytes()); // flags
        ram[d0_off + 14..d0_off + 16].copy_from_slice(&1u16.to_le_bytes()); // next

        // Descriptor 1: 64-byte packet payload (read-only)
        let d1_off = 0x1000 + 16;
        let payload_addr: u64 = dram_base + 0x5000;
        ram[d1_off..d1_off + 8].copy_from_slice(&payload_addr.to_le_bytes());
        ram[d1_off + 8..d1_off + 12].copy_from_slice(&64u32.to_le_bytes());
        ram[d1_off + 12..d1_off + 14].copy_from_slice(&0u16.to_le_bytes()); // no next, no write
        ram[d1_off + 14..d1_off + 16].copy_from_slice(&0u16.to_le_bytes());

        // Set up avail ring: flags=0, idx=1, ring[0]=0
        let avail_off = 0x2000usize;
        ram[avail_off..avail_off + 2].copy_from_slice(&0u16.to_le_bytes()); // flags
        ram[avail_off + 2..avail_off + 4].copy_from_slice(&1u16.to_le_bytes()); // idx
        ram[avail_off + 4..avail_off + 6].copy_from_slice(&0u16.to_le_bytes()); // ring[0] = desc 0

        // Notify TX queue
        net.write(QUEUE_NOTIFY, 1);
        assert!(net.needs_processing());

        // Process
        net.process_queues(&mut ram, dram_base);

        // Check used ring: idx should be 1
        let used_off = 0x3000usize;
        let used_idx = u16::from_le_bytes([ram[used_off + 2], ram[used_off + 3]]);
        assert_eq!(used_idx, 1, "Used ring index should advance");

        // Check interrupt
        assert!(net.has_interrupt(), "Interrupt should be pending after TX");
    }
}
