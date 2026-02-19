/// VirtIO Entropy (RNG) Device — provides randomness to the guest
///
/// VirtIO device type 4. The guest submits buffers on virtqueue 0,
/// and the device fills them with random bytes.
///
/// This uses the host's OS RNG via `getrandom`.
use std::io::Read;

const VIRTIO_MAGIC: u32 = 0x7472_6976; // "virt"
const VIRTIO_VERSION: u32 = 2; // non-legacy MMIO
const DEVICE_ID: u32 = 4; // entropy source
const VENDOR_ID: u32 = 0x554D_4356; // "UMCV"

// Device status bits
const STATUS_ACKNOWLEDGE: u32 = 1;
const STATUS_DRIVER: u32 = 2;
const STATUS_FEATURES_OK: u32 = 8;
const STATUS_DRIVER_OK: u32 = 4;

pub struct VirtioRng {
    // Device status
    status: u32,
    // Selected queue (only queue 0)
    queue_sel: u32,
    // Queue 0 descriptor area (physical address)
    queue_desc: u64,
    // Queue 0 driver (avail) area
    queue_driver: u64,
    // Queue 0 device (used) area
    queue_device: u64,
    // Queue size (max descriptors)
    queue_num: u32,
    // Queue ready flag
    queue_ready: bool,
    // Last seen avail index
    last_avail_idx: u16,
    // Interrupt status
    interrupt_status: u32,
    // Guest feature selection page
    guest_features_sel: u32,
    // Guest features
    #[allow(dead_code)]
    guest_features: u64,
    // Driver features selection page
    driver_features_sel: u32,
    // Driver features
    driver_features: u64,
    // Queue notify flag
    notify: bool,
}

impl Default for VirtioRng {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioRng {
    pub fn new() -> Self {
        Self {
            status: 0,
            queue_sel: 0,
            queue_desc: 0,
            queue_driver: 0,
            queue_device: 0,
            queue_num: 256,
            queue_ready: false,
            last_avail_idx: 0,
            interrupt_status: 0,
            guest_features_sel: 0,
            guest_features: 0,
            driver_features_sel: 0,
            driver_features: 0,
            notify: false,
        }
    }

    pub fn read(&self, offset: u64) -> u32 {
        match offset {
            0x000 => VIRTIO_MAGIC,
            0x004 => VIRTIO_VERSION,
            0x008 => DEVICE_ID,
            0x00C => VENDOR_ID,
            0x010 => {
                // DeviceFeatures — no special features for RNG
                if self.guest_features_sel == 0 {
                    // Bit 0 would be VIRTIO_F_RING_INDIRECT_DESC etc; keep it simple
                    0
                } else {
                    0
                }
            }
            0x034 => self.queue_num.min(256), // QueueNumMax
            0x044 => {
                if self.queue_ready {
                    1
                } else {
                    0
                }
            }
            0x060 => self.interrupt_status,
            0x070 => self.status,
            0x0FC => 0x02, // ConfigGeneration
            _ => 0,
        }
    }

    pub fn write(&mut self, offset: u64, val: u64) {
        match offset {
            0x014 => self.guest_features_sel = val as u32,
            0x020 => {
                // DriverFeatures
                if self.driver_features_sel == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                } else {
                    self.driver_features = (self.driver_features & 0x0000_0000_FFFF_FFFF)
                        | ((val & 0xFFFF_FFFF) << 32);
                }
            }
            0x024 => self.driver_features_sel = val as u32,
            0x030 => self.queue_sel = val as u32,
            0x038 => self.queue_num = (val as u32).min(256),
            0x044 => self.queue_ready = val & 1 != 0,
            0x050 => {
                // QueueNotify
                if val == 0 {
                    self.notify = true;
                }
            }
            0x064 => self.interrupt_status &= !(val as u32), // InterruptACK
            0x070 => {
                // Status
                self.status = val as u32;
                if val == 0 {
                    self.reset();
                }
            }
            0x080 => {
                self.queue_desc = (self.queue_desc & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
            }
            0x084 => {
                self.queue_desc =
                    (self.queue_desc & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            0x090 => {
                self.queue_driver =
                    (self.queue_driver & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
            }
            0x094 => {
                self.queue_driver =
                    (self.queue_driver & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            0x0A0 => {
                self.queue_device =
                    (self.queue_device & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
            }
            0x0A4 => {
                self.queue_device =
                    (self.queue_device & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.status = 0;
        self.queue_ready = false;
        self.last_avail_idx = 0;
        self.interrupt_status = 0;
        self.notify = false;
        self.queue_desc = 0;
        self.queue_driver = 0;
        self.queue_device = 0;
        self.driver_features = 0;
    }

    pub fn has_interrupt(&self) -> bool {
        self.interrupt_status != 0
    }

    pub fn needs_processing(&self) -> bool {
        self.notify
            && self.queue_ready
            && (self.status & STATUS_DRIVER_OK) != 0
            && (self.status & STATUS_ACKNOWLEDGE) != 0
            && (self.status & STATUS_DRIVER) != 0
            && (self.status & STATUS_FEATURES_OK) != 0
    }

    /// Process the virtqueue: fill guest buffers with random bytes
    pub fn process_queue(&mut self, ram: &mut [u8], dram_base: u64) {
        if !self.needs_processing() {
            return;
        }
        self.notify = false;

        let desc_base = self.queue_desc;
        let avail_base = self.queue_driver;
        let used_base = self.queue_device;
        let queue_size = self.queue_num as u16;

        // Read avail index
        let avail_idx_off = (avail_base - dram_base + 2) as usize;
        if avail_idx_off + 2 > ram.len() {
            return;
        }
        let avail_idx = u16::from_le_bytes([ram[avail_idx_off], ram[avail_idx_off + 1]]);

        let mut used_count = 0u16;

        while self.last_avail_idx != avail_idx {
            let ring_idx = (self.last_avail_idx % queue_size) as usize;
            let avail_ring_off = (avail_base - dram_base + 4 + ring_idx as u64 * 2) as usize;
            if avail_ring_off + 2 > ram.len() {
                break;
            }
            let desc_idx =
                u16::from_le_bytes([ram[avail_ring_off], ram[avail_ring_off + 1]]) as u64;

            // Read descriptor
            let desc_off = (desc_base - dram_base + desc_idx * 16) as usize;
            if desc_off + 16 > ram.len() {
                break;
            }
            let buf_addr = u64::from_le_bytes(ram[desc_off..desc_off + 8].try_into().unwrap());
            let buf_len = u32::from_le_bytes(ram[desc_off + 8..desc_off + 12].try_into().unwrap());
            let flags = u16::from_le_bytes(ram[desc_off + 12..desc_off + 14].try_into().unwrap());

            // Flag bit 1 = VIRTQ_DESC_F_WRITE (device writes to this buffer)
            if flags & 2 != 0 {
                let ram_off = (buf_addr - dram_base) as usize;
                let len = buf_len as usize;
                if ram_off + len <= ram.len() {
                    // Fill with random bytes from OS
                    let buf = &mut ram[ram_off..ram_off + len];
                    let _ = getrandom(buf);
                }
            }

            // Write used ring entry
            let used_idx_off = (used_base - dram_base + 2) as usize;
            if used_idx_off + 2 > ram.len() {
                break;
            }
            let current_used_idx = u16::from_le_bytes([ram[used_idx_off], ram[used_idx_off + 1]]);
            let used_ring_entry =
                (used_base - dram_base + 4 + (current_used_idx % queue_size) as u64 * 8) as usize;
            if used_ring_entry + 8 > ram.len() {
                break;
            }
            ram[used_ring_entry..used_ring_entry + 4]
                .copy_from_slice(&(desc_idx as u32).to_le_bytes());
            ram[used_ring_entry + 4..used_ring_entry + 8].copy_from_slice(&(buf_len).to_le_bytes());

            // Increment used index
            let new_used_idx = current_used_idx.wrapping_add(1);
            ram[used_idx_off..used_idx_off + 2].copy_from_slice(&new_used_idx.to_le_bytes());

            self.last_avail_idx = self.last_avail_idx.wrapping_add(1);
            used_count += 1;
        }

        if used_count > 0 {
            self.interrupt_status |= 1; // Used buffer notification
        }
    }
}

/// Fill buffer with random bytes using the OS RNG
fn getrandom(buf: &mut [u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::open("/dev/urandom")?;
    file.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtio_rng_magic_and_id() {
        let rng = VirtioRng::new();
        assert_eq!(rng.read(0x000), VIRTIO_MAGIC);
        assert_eq!(rng.read(0x004), VIRTIO_VERSION);
        assert_eq!(rng.read(0x008), 4); // device type 4 = entropy
        assert_eq!(rng.read(0x00C), VENDOR_ID);
    }

    #[test]
    fn test_virtio_rng_status_lifecycle() {
        let mut rng = VirtioRng::new();
        assert_eq!(rng.read(0x070), 0);

        // Driver acknowledges
        rng.write(0x070, STATUS_ACKNOWLEDGE as u64);
        assert_eq!(rng.read(0x070), STATUS_ACKNOWLEDGE);

        // Driver says it knows the device
        rng.write(0x070, (STATUS_ACKNOWLEDGE | STATUS_DRIVER) as u64);
        assert_eq!(rng.read(0x070), STATUS_ACKNOWLEDGE | STATUS_DRIVER);

        // Reset
        rng.write(0x070, 0);
        assert_eq!(rng.read(0x070), 0);
    }

    #[test]
    fn test_virtio_rng_queue_setup() {
        let mut rng = VirtioRng::new();
        // Select queue 0
        rng.write(0x030, 0);
        assert_eq!(rng.read(0x034), 256); // QueueNumMax

        // Set queue size
        rng.write(0x038, 128);
        // Set queue descriptors address
        rng.write(0x080, 0x1000);
        rng.write(0x084, 0);
        // Set queue ready
        rng.write(0x044, 1);
        assert_eq!(rng.read(0x044), 1);
    }

    #[test]
    fn test_virtio_rng_interrupt_ack() {
        let mut rng = VirtioRng::new();
        rng.interrupt_status = 1;
        assert!(rng.has_interrupt());
        rng.write(0x064, 1); // ACK
        assert!(!rng.has_interrupt());
    }
}
