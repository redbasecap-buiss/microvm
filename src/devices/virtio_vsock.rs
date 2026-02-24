/// VirtIO MMIO Socket (vsock) Device (v2)
///
/// Implements a VirtIO socket device (device type 19) for host-guest
/// communication via AF_VSOCK. The device uses three virtqueues:
/// RX (queue 0), TX (queue 1), and Event (queue 2).
///
/// The guest is assigned a CID (Context ID). CID 2 is conventionally
/// the host, CID 3+ are guests. Without a host backend, the device
/// accepts connections and silently drops packets.
///
/// Protocol: VirtIO vsock uses a simple header-based transport with
/// stream (SOCK_STREAM) and datagram (SOCK_DGRAM) socket types.
///
/// Reference: VirtIO spec v1.2, sections 2, 4.2 (MMIO), 5.10 (socket device).
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
const CONFIG_BASE: u64 = 0x100;

// VirtIO device status bits
const STATUS_DRIVER_OK: u32 = 4;

// Feature bits
const VIRTIO_VSOCK_F_SEQPACKET: u64 = 1 << 1; // SOCK_SEQPACKET support

// Virtqueue descriptor flags
const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

const QUEUE_SIZE: u32 = 256;
const NUM_QUEUES: usize = 3; // RX (0), TX (1), Event (2)

/// Default guest CID
const GUEST_CID: u64 = 3;

/// vsock packet operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
enum VsockOp {
    Invalid = 0,
    Request = 1,
    Response = 2,
    Rst = 3,
    Shutdown = 4,
    Rw = 5,
    CreditUpdate = 6,
    CreditRequest = 7,
}

impl VsockOp {
    fn from_u16(v: u16) -> Self {
        match v {
            1 => VsockOp::Request,
            2 => VsockOp::Response,
            3 => VsockOp::Rst,
            4 => VsockOp::Shutdown,
            5 => VsockOp::Rw,
            6 => VsockOp::CreditUpdate,
            7 => VsockOp::CreditRequest,
            _ => VsockOp::Invalid,
        }
    }
}

/// vsock packet header (44 bytes)
#[derive(Debug, Clone)]
struct VsockHeader {
    src_cid: u64,
    dst_cid: u64,
    src_port: u32,
    dst_port: u32,
    len: u32,
    r#type: u16,
    op: VsockOp,
    flags: u32,
    buf_alloc: u32,
    fwd_cnt: u32,
}

const VSOCK_HEADER_SIZE: usize = 44;

impl VsockHeader {
    fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < VSOCK_HEADER_SIZE {
            return None;
        }
        Some(Self {
            src_cid: u64::from_le_bytes(data[0..8].try_into().ok()?),
            dst_cid: u64::from_le_bytes(data[8..16].try_into().ok()?),
            src_port: u32::from_le_bytes(data[16..20].try_into().ok()?),
            dst_port: u32::from_le_bytes(data[20..24].try_into().ok()?),
            len: u32::from_le_bytes(data[24..28].try_into().ok()?),
            r#type: u16::from_le_bytes(data[28..30].try_into().ok()?),
            op: VsockOp::from_u16(u16::from_le_bytes(data[30..32].try_into().ok()?)),
            flags: u32::from_le_bytes(data[32..36].try_into().ok()?),
            buf_alloc: u32::from_le_bytes(data[36..40].try_into().ok()?),
            fwd_cnt: u32::from_le_bytes(data[40..44].try_into().ok()?),
        })
    }

    fn to_bytes(&self) -> [u8; VSOCK_HEADER_SIZE] {
        let mut buf = [0u8; VSOCK_HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.src_cid.to_le_bytes());
        buf[8..16].copy_from_slice(&self.dst_cid.to_le_bytes());
        buf[16..20].copy_from_slice(&self.src_port.to_le_bytes());
        buf[20..24].copy_from_slice(&self.dst_port.to_le_bytes());
        buf[24..28].copy_from_slice(&self.len.to_le_bytes());
        buf[28..30].copy_from_slice(&self.r#type.to_le_bytes());
        buf[30..32].copy_from_slice(&(self.op as u16).to_le_bytes());
        buf[32..36].copy_from_slice(&self.flags.to_le_bytes());
        buf[36..40].copy_from_slice(&self.buf_alloc.to_le_bytes());
        buf[40..44].copy_from_slice(&self.fwd_cnt.to_le_bytes());
        buf
    }
}

/// A tracked vsock connection
#[derive(Debug, Clone)]
struct VsockConnection {
    src_cid: u64,
    src_port: u32,
    dst_cid: u64,
    dst_port: u32,
}

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

/// VirtIO MMIO Socket Device
pub struct VirtioVsock {
    // MMIO state
    device_features_sel: u32,
    driver_features: u64,
    driver_features_sel: u32,
    queue_sel: u32,
    queues: [Virtqueue; NUM_QUEUES],
    interrupt_status: u32,
    status: u32,
    // Device-specific
    guest_cid: u64,
    /// Pending TX packets from guest (header + payload)
    tx_packets: Vec<(VsockHeader, Vec<u8>)>,
    /// Pending RX packets to deliver to guest
    rx_queue: Vec<(VsockHeader, Vec<u8>)>,
    /// Pending event notifications
    event_queue: Vec<u32>,
    /// Active connections
    connections: Vec<VsockConnection>,
    /// Flag for needs_processing
    notify_pending: bool,
}

impl Default for VirtioVsock {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioVsock {
    pub fn new() -> Self {
        Self {
            device_features_sel: 0,
            driver_features: 0,
            driver_features_sel: 0,
            queue_sel: 0,
            queues: [Virtqueue::new(), Virtqueue::new(), Virtqueue::new()],
            interrupt_status: 0,
            status: 0,
            guest_cid: GUEST_CID,
            tx_packets: Vec::new(),
            rx_queue: Vec::new(),
            event_queue: Vec::new(),
            connections: Vec::new(),
            notify_pending: false,
        }
    }

    pub fn read(&self, offset: u64) -> u32 {
        match offset {
            MAGIC_VALUE => 0x7472_6976, // "virt"
            VERSION => 2,               // VirtIO MMIO v2
            DEVICE_ID => 19,            // Socket device
            VENDOR_ID => 0x554D_4552,   // "REMU"
            DEVICE_FEATURES => {
                let page = self.device_features_sel;
                match page {
                    0 => VIRTIO_VSOCK_F_SEQPACKET as u32,
                    1 => 0x1, // VIRTIO_F_VERSION_1
                    _ => 0,
                }
            }
            QUEUE_NUM_MAX => QUEUE_SIZE,
            QUEUE_READY => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    self.queues[sel].ready as u32
                } else {
                    0
                }
            }
            INTERRUPT_STATUS => self.interrupt_status,
            STATUS => self.status,
            CONFIG_GENERATION => 0,
            // Config space: guest_cid (8 bytes at offset 0x100)
            CONFIG_BASE => self.guest_cid as u32,
            0x104 => (self.guest_cid >> 32) as u32,
            _ => 0,
        }
    }

    pub fn write(&mut self, offset: u64, val: u64) {
        let val32 = val as u32;
        match offset {
            DEVICE_FEATURES_SEL => self.device_features_sel = val32,
            DRIVER_FEATURES => {
                let shift = self.driver_features_sel * 32;
                let mask = !((0xFFFF_FFFF_u64) << shift);
                self.driver_features = (self.driver_features & mask) | ((val32 as u64) << shift);
            }
            DRIVER_FEATURES_SEL => self.driver_features_sel = val32,
            QUEUE_SEL => self.queue_sel = val32,
            QUEUE_NUM => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    self.queues[sel].num = val32;
                }
            }
            QUEUE_READY => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    self.queues[sel].ready = val32 != 0;
                }
            }
            QUEUE_NOTIFY => {
                self.notify_pending = true;
            }
            INTERRUPT_ACK => {
                self.interrupt_status &= !val32;
            }
            STATUS => {
                self.status = val32;
                if val32 == 0 {
                    self.reset();
                }
            }
            QUEUE_DESC_LOW => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    let q = &mut self.queues[sel];
                    q.desc_addr = (q.desc_addr & 0xFFFF_FFFF_0000_0000) | val32 as u64;
                }
            }
            QUEUE_DESC_HIGH => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    let q = &mut self.queues[sel];
                    q.desc_addr = (q.desc_addr & 0x0000_0000_FFFF_FFFF) | ((val32 as u64) << 32);
                }
            }
            QUEUE_AVAIL_LOW => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    let q = &mut self.queues[sel];
                    q.avail_addr = (q.avail_addr & 0xFFFF_FFFF_0000_0000) | val32 as u64;
                }
            }
            QUEUE_AVAIL_HIGH => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    let q = &mut self.queues[sel];
                    q.avail_addr = (q.avail_addr & 0x0000_0000_FFFF_FFFF) | ((val32 as u64) << 32);
                }
            }
            QUEUE_USED_LOW => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    let q = &mut self.queues[sel];
                    q.used_addr = (q.used_addr & 0xFFFF_FFFF_0000_0000) | val32 as u64;
                }
            }
            QUEUE_USED_HIGH => {
                let sel = self.queue_sel as usize;
                if sel < NUM_QUEUES {
                    let q = &mut self.queues[sel];
                    q.used_addr = (q.used_addr & 0x0000_0000_FFFF_FFFF) | ((val32 as u64) << 32);
                }
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.device_features_sel = 0;
        self.driver_features = 0;
        self.driver_features_sel = 0;
        self.queue_sel = 0;
        for q in &mut self.queues {
            *q = Virtqueue::new();
        }
        self.interrupt_status = 0;
        self.status = 0;
        self.tx_packets.clear();
        self.rx_queue.clear();
        self.event_queue.clear();
        self.connections.clear();
        self.notify_pending = false;
    }

    pub fn has_interrupt(&self) -> bool {
        self.interrupt_status != 0
    }

    pub fn needs_processing(&self) -> bool {
        self.notify_pending
    }

    /// Process TX queue: read packets sent by guest
    pub fn process_queues(&mut self, ram: &mut [u8], dram_base: u64) {
        self.notify_pending = false;

        if self.status & STATUS_DRIVER_OK == 0 {
            return;
        }

        // Process TX queue (queue 1): guest sends packets
        self.process_tx(ram, dram_base);

        // Process received packets: generate responses
        self.handle_tx_packets();

        // Deliver RX queue (queue 0): send responses to guest
        self.deliver_rx(ram, dram_base);

        // Deliver events (queue 2)
        self.deliver_events(ram, dram_base);
    }

    fn process_tx(&mut self, ram: &mut [u8], dram_base: u64) {
        let q = &mut self.queues[1]; // TX queue
        if !q.ready || q.num == 0 {
            return;
        }

        let desc_base = q.desc_addr.wrapping_sub(dram_base) as usize;
        let avail_base = q.avail_addr.wrapping_sub(dram_base) as usize;
        let used_base = q.used_addr.wrapping_sub(dram_base) as usize;

        if avail_base + 4 > ram.len() || used_base + 4 > ram.len() {
            return;
        }

        let avail_idx = u16::from_le_bytes([ram[avail_base + 2], ram[avail_base + 3]]);

        let mut packets = Vec::new();

        while q.last_avail_idx != avail_idx {
            let ring_off = avail_base + 4 + (q.last_avail_idx as usize % q.num as usize) * 2;
            if ring_off + 2 > ram.len() {
                break;
            }
            let head = u16::from_le_bytes([ram[ring_off], ram[ring_off + 1]]);

            // Gather descriptor chain data
            let mut data = Vec::new();
            let mut idx = head;
            let mut total_len = 0u32;
            loop {
                let d_off = desc_base + idx as usize * 16;
                if d_off + 16 > ram.len() {
                    break;
                }
                let addr = u64::from_le_bytes(ram[d_off..d_off + 8].try_into().unwrap());
                let len = u32::from_le_bytes(ram[d_off + 8..d_off + 12].try_into().unwrap());
                let flags = u16::from_le_bytes(ram[d_off + 12..d_off + 14].try_into().unwrap());

                let phys = addr.wrapping_sub(dram_base) as usize;
                if phys + len as usize <= ram.len() && flags & VRING_DESC_F_WRITE == 0 {
                    data.extend_from_slice(&ram[phys..phys + len as usize]);
                }
                total_len += len;

                if flags & VRING_DESC_F_NEXT == 0 {
                    break;
                }
                let next = u16::from_le_bytes(ram[d_off + 14..d_off + 16].try_into().unwrap());
                idx = next;
            }

            // Parse vsock header
            if let Some(hdr) = VsockHeader::from_bytes(&data) {
                let payload = if data.len() > VSOCK_HEADER_SIZE {
                    data[VSOCK_HEADER_SIZE..].to_vec()
                } else {
                    Vec::new()
                };
                packets.push((hdr, payload));
            }

            // Write used ring entry
            let used_idx_off = used_base + 2;
            if used_idx_off + 2 <= ram.len() {
                let used_idx = u16::from_le_bytes([ram[used_idx_off], ram[used_idx_off + 1]]);
                let used_ring_off = used_base + 4 + (used_idx as usize % q.num as usize) * 8;
                if used_ring_off + 8 <= ram.len() {
                    ram_write_u32(ram, used_ring_off, head as u32);
                    ram_write_u32(ram, used_ring_off + 4, total_len);
                }
                let new_idx = used_idx.wrapping_add(1);
                ram[used_idx_off] = new_idx as u8;
                ram[used_idx_off + 1] = (new_idx >> 8) as u8;
            }

            q.last_avail_idx = q.last_avail_idx.wrapping_add(1);
        }

        self.tx_packets.extend(packets);
        if !self.tx_packets.is_empty() {
            self.interrupt_status |= 1; // Used buffer notification
        }
    }

    fn handle_tx_packets(&mut self) {
        let packets: Vec<_> = self.tx_packets.drain(..).collect();
        for (hdr, _payload) in packets {
            match hdr.op {
                VsockOp::Request => {
                    // Guest is connecting — send RST (no host backend)
                    let rst = VsockHeader {
                        src_cid: hdr.dst_cid,
                        dst_cid: hdr.src_cid,
                        src_port: hdr.dst_port,
                        dst_port: hdr.src_port,
                        len: 0,
                        r#type: 1, // VIRTIO_VSOCK_TYPE_STREAM
                        op: VsockOp::Rst,
                        flags: 0,
                        buf_alloc: 0,
                        fwd_cnt: 0,
                    };
                    self.rx_queue.push((rst, Vec::new()));
                }
                VsockOp::Shutdown | VsockOp::Rst => {
                    // Remove connection if exists
                    self.connections.retain(|c| {
                        !(c.src_cid == hdr.src_cid
                            && c.src_port == hdr.src_port
                            && c.dst_cid == hdr.dst_cid
                            && c.dst_port == hdr.dst_port)
                    });
                    // Acknowledge with RST
                    let rst = VsockHeader {
                        src_cid: hdr.dst_cid,
                        dst_cid: hdr.src_cid,
                        src_port: hdr.dst_port,
                        dst_port: hdr.src_port,
                        len: 0,
                        r#type: 1,
                        op: VsockOp::Rst,
                        flags: 0,
                        buf_alloc: 0,
                        fwd_cnt: 0,
                    };
                    self.rx_queue.push((rst, Vec::new()));
                }
                VsockOp::CreditRequest => {
                    // Respond with credit update
                    let credit = VsockHeader {
                        src_cid: hdr.dst_cid,
                        dst_cid: hdr.src_cid,
                        src_port: hdr.dst_port,
                        dst_port: hdr.src_port,
                        len: 0,
                        r#type: 1,
                        op: VsockOp::CreditUpdate,
                        flags: 0,
                        buf_alloc: 65536, // 64K buffer
                        fwd_cnt: 0,
                    };
                    self.rx_queue.push((credit, Vec::new()));
                }
                VsockOp::Rw | VsockOp::Response | VsockOp::CreditUpdate => {
                    // Data/response/credit — silently consume (no host backend)
                }
                VsockOp::Invalid => {}
            }
        }
    }

    fn deliver_rx(&mut self, ram: &mut [u8], dram_base: u64) {
        let q = &mut self.queues[0]; // RX queue
        if !q.ready || q.num == 0 || self.rx_queue.is_empty() {
            return;
        }

        let desc_base = q.desc_addr.wrapping_sub(dram_base) as usize;
        let avail_base = q.avail_addr.wrapping_sub(dram_base) as usize;
        let used_base = q.used_addr.wrapping_sub(dram_base) as usize;

        if avail_base + 4 > ram.len() || used_base + 4 > ram.len() {
            return;
        }

        let avail_idx = u16::from_le_bytes([ram[avail_base + 2], ram[avail_base + 3]]);

        while q.last_avail_idx != avail_idx && !self.rx_queue.is_empty() {
            let ring_off = avail_base + 4 + (q.last_avail_idx as usize % q.num as usize) * 2;
            if ring_off + 2 > ram.len() {
                break;
            }
            let head = u16::from_le_bytes([ram[ring_off], ram[ring_off + 1]]);

            let (hdr, payload) = self.rx_queue.remove(0);
            let pkt_data = hdr.to_bytes();

            // Write packet to descriptor chain (writable buffers)
            let mut idx = head;
            let mut written = 0usize;
            let total_data_len = VSOCK_HEADER_SIZE + payload.len();
            loop {
                let d_off = desc_base + idx as usize * 16;
                if d_off + 16 > ram.len() {
                    break;
                }
                let addr = u64::from_le_bytes(ram[d_off..d_off + 8].try_into().unwrap());
                let len = u32::from_le_bytes(ram[d_off + 8..d_off + 12].try_into().unwrap());
                let flags = u16::from_le_bytes(ram[d_off + 12..d_off + 14].try_into().unwrap());

                if flags & VRING_DESC_F_WRITE != 0 {
                    let phys = addr.wrapping_sub(dram_base) as usize;
                    let buf_len = len as usize;
                    if phys + buf_len <= ram.len() {
                        // Write header + payload into buffer
                        let mut offset = 0;
                        while written < total_data_len && offset < buf_len {
                            let byte = if written < VSOCK_HEADER_SIZE {
                                pkt_data[written]
                            } else {
                                payload[written - VSOCK_HEADER_SIZE]
                            };
                            ram[phys + offset] = byte;
                            written += 1;
                            offset += 1;
                        }
                    }
                }

                if flags & VRING_DESC_F_NEXT == 0 {
                    break;
                }
                let next = u16::from_le_bytes(ram[d_off + 14..d_off + 16].try_into().unwrap());
                idx = next;
            }

            // Write used ring
            let used_idx_off = used_base + 2;
            if used_idx_off + 2 <= ram.len() {
                let used_idx = u16::from_le_bytes([ram[used_idx_off], ram[used_idx_off + 1]]);
                let used_ring_off = used_base + 4 + (used_idx as usize % q.num as usize) * 8;
                if used_ring_off + 8 <= ram.len() {
                    ram_write_u32(ram, used_ring_off, head as u32);
                    ram_write_u32(ram, used_ring_off + 4, written as u32);
                }
                let new_idx = used_idx.wrapping_add(1);
                ram[used_idx_off] = new_idx as u8;
                ram[used_idx_off + 1] = (new_idx >> 8) as u8;
            }

            q.last_avail_idx = q.last_avail_idx.wrapping_add(1);
        }

        if !self.rx_queue.is_empty() || written_any_rx(q) {
            self.interrupt_status |= 1;
        }
    }

    fn deliver_events(&mut self, ram: &mut [u8], dram_base: u64) {
        let q = &mut self.queues[2]; // Event queue
        if !q.ready || q.num == 0 || self.event_queue.is_empty() {
            return;
        }

        let desc_base = q.desc_addr.wrapping_sub(dram_base) as usize;
        let avail_base = q.avail_addr.wrapping_sub(dram_base) as usize;
        let used_base = q.used_addr.wrapping_sub(dram_base) as usize;

        if avail_base + 4 > ram.len() || used_base + 4 > ram.len() {
            return;
        }

        let avail_idx = u16::from_le_bytes([ram[avail_base + 2], ram[avail_base + 3]]);

        while q.last_avail_idx != avail_idx && !self.event_queue.is_empty() {
            let ring_off = avail_base + 4 + (q.last_avail_idx as usize % q.num as usize) * 2;
            if ring_off + 2 > ram.len() {
                break;
            }
            let head = u16::from_le_bytes([ram[ring_off], ram[ring_off + 1]]);

            let event_id = self.event_queue.remove(0);

            // Write event to first writable descriptor
            let d_off = desc_base + head as usize * 16;
            if d_off + 16 <= ram.len() {
                let addr = u64::from_le_bytes(ram[d_off..d_off + 8].try_into().unwrap());
                let flags = u16::from_le_bytes(ram[d_off + 12..d_off + 14].try_into().unwrap());
                if flags & VRING_DESC_F_WRITE != 0 {
                    let phys = addr.wrapping_sub(dram_base) as usize;
                    if phys + 4 <= ram.len() {
                        ram_write_u32(ram, phys, event_id);
                    }
                }
            }

            // Write used ring
            let used_idx_off = used_base + 2;
            if used_idx_off + 2 <= ram.len() {
                let used_idx = u16::from_le_bytes([ram[used_idx_off], ram[used_idx_off + 1]]);
                let used_ring_off = used_base + 4 + (used_idx as usize % q.num as usize) * 8;
                if used_ring_off + 8 <= ram.len() {
                    ram_write_u32(ram, used_ring_off, head as u32);
                    ram_write_u32(ram, used_ring_off + 4, 4); // 4 bytes written
                }
                let new_idx = used_idx.wrapping_add(1);
                ram[used_idx_off] = new_idx as u8;
                ram[used_idx_off + 1] = (new_idx >> 8) as u8;
            }

            q.last_avail_idx = q.last_avail_idx.wrapping_add(1);
        }

        if !self.event_queue.is_empty() {
            self.interrupt_status |= 1;
        }
    }
}

/// Helper: did we deliver any RX? (always true if we entered the loop)
fn written_any_rx(_q: &Virtqueue) -> bool {
    true
}

fn ram_write_u32(ram: &mut [u8], offset: usize, val: u32) {
    if offset + 4 <= ram.len() {
        ram[offset] = val as u8;
        ram[offset + 1] = (val >> 8) as u8;
        ram[offset + 2] = (val >> 16) as u8;
        ram[offset + 3] = (val >> 24) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsock_magic_and_device_id() {
        let dev = VirtioVsock::new();
        assert_eq!(dev.read(MAGIC_VALUE), 0x7472_6976);
        assert_eq!(dev.read(VERSION), 2);
        assert_eq!(dev.read(DEVICE_ID), 19);
        assert_eq!(dev.read(VENDOR_ID), 0x554D_4552);
    }

    #[test]
    fn test_vsock_config_guest_cid() {
        let dev = VirtioVsock::new();
        assert_eq!(dev.read(CONFIG_BASE), GUEST_CID as u32);
        assert_eq!(dev.read(0x104), (GUEST_CID >> 32) as u32);
    }

    #[test]
    fn test_vsock_features() {
        let mut dev = VirtioVsock::new();
        // Page 0: device-specific features
        dev.write(DEVICE_FEATURES_SEL, 0);
        let f0 = dev.read(DEVICE_FEATURES);
        assert_ne!(f0 & VIRTIO_VSOCK_F_SEQPACKET as u32, 0);
        // Page 1: VIRTIO_F_VERSION_1
        dev.write(DEVICE_FEATURES_SEL, 1);
        let f1 = dev.read(DEVICE_FEATURES);
        assert_eq!(f1 & 1, 1);
    }

    #[test]
    fn test_vsock_queue_setup() {
        let mut dev = VirtioVsock::new();
        // Select queue 0 (RX)
        dev.write(QUEUE_SEL, 0);
        assert_eq!(dev.read(QUEUE_NUM_MAX), QUEUE_SIZE);
        dev.write(QUEUE_NUM, 128);
        dev.write(QUEUE_DESC_LOW, 0x1000);
        dev.write(QUEUE_DESC_HIGH, 0);
        dev.write(QUEUE_AVAIL_LOW, 0x2000);
        dev.write(QUEUE_AVAIL_HIGH, 0);
        dev.write(QUEUE_USED_LOW, 0x3000);
        dev.write(QUEUE_USED_HIGH, 0);
        dev.write(QUEUE_READY, 1);
        assert_eq!(dev.read(QUEUE_READY), 1);
    }

    #[test]
    fn test_vsock_status_and_reset() {
        let mut dev = VirtioVsock::new();
        dev.write(STATUS, 1); // ACKNOWLEDGE
        assert_eq!(dev.read(STATUS), 1);
        dev.write(STATUS, 3); // ACKNOWLEDGE | DRIVER
        assert_eq!(dev.read(STATUS), 3);
        // Reset
        dev.write(STATUS, 0);
        assert_eq!(dev.read(STATUS), 0);
    }

    #[test]
    fn test_vsock_interrupt_ack() {
        let mut dev = VirtioVsock::new();
        dev.interrupt_status = 0x3;
        assert!(dev.has_interrupt());
        dev.write(INTERRUPT_ACK, 0x1);
        assert_eq!(dev.interrupt_status, 0x2);
        dev.write(INTERRUPT_ACK, 0x2);
        assert!(!dev.has_interrupt());
    }

    #[test]
    fn test_vsock_driver_features() {
        let mut dev = VirtioVsock::new();
        dev.write(DRIVER_FEATURES_SEL, 0);
        dev.write(DRIVER_FEATURES, 0x02); // SEQPACKET
        assert_eq!(dev.driver_features & 0xFF, 0x02);
        dev.write(DRIVER_FEATURES_SEL, 1);
        dev.write(DRIVER_FEATURES, 0x01); // VERSION_1
        assert_eq!(dev.driver_features >> 32, 0x01);
    }

    #[test]
    fn test_vsock_three_queues() {
        let mut dev = VirtioVsock::new();
        // All three queues should be configurable
        for q in 0..3 {
            dev.write(QUEUE_SEL, q);
            assert_eq!(dev.read(QUEUE_NUM_MAX), QUEUE_SIZE);
            dev.write(QUEUE_NUM, 64);
            dev.write(QUEUE_READY, 1);
            assert_eq!(dev.read(QUEUE_READY), 1);
        }
        // Queue 3+ should not exist
        dev.write(QUEUE_SEL, 3);
        dev.write(QUEUE_READY, 1);
        assert_eq!(dev.read(QUEUE_READY), 0);
    }

    #[test]
    fn test_vsock_header_roundtrip() {
        let hdr = VsockHeader {
            src_cid: 3,
            dst_cid: 2,
            src_port: 1234,
            dst_port: 5678,
            len: 100,
            r#type: 1,
            op: VsockOp::Request,
            flags: 0,
            buf_alloc: 65536,
            fwd_cnt: 42,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), VSOCK_HEADER_SIZE);
        let decoded = VsockHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.src_cid, 3);
        assert_eq!(decoded.dst_cid, 2);
        assert_eq!(decoded.src_port, 1234);
        assert_eq!(decoded.dst_port, 5678);
        assert_eq!(decoded.len, 100);
        assert_eq!(decoded.r#type, 1);
        assert_eq!(decoded.op, VsockOp::Request);
        assert_eq!(decoded.buf_alloc, 65536);
        assert_eq!(decoded.fwd_cnt, 42);
    }

    #[test]
    fn test_vsock_header_too_short() {
        let data = [0u8; 10];
        assert!(VsockHeader::from_bytes(&data).is_none());
    }

    #[test]
    fn test_vsock_op_from_u16() {
        assert_eq!(VsockOp::from_u16(1), VsockOp::Request);
        assert_eq!(VsockOp::from_u16(2), VsockOp::Response);
        assert_eq!(VsockOp::from_u16(3), VsockOp::Rst);
        assert_eq!(VsockOp::from_u16(4), VsockOp::Shutdown);
        assert_eq!(VsockOp::from_u16(5), VsockOp::Rw);
        assert_eq!(VsockOp::from_u16(6), VsockOp::CreditUpdate);
        assert_eq!(VsockOp::from_u16(7), VsockOp::CreditRequest);
        assert_eq!(VsockOp::from_u16(99), VsockOp::Invalid);
    }

    #[test]
    fn test_vsock_needs_processing() {
        let mut dev = VirtioVsock::new();
        assert!(!dev.needs_processing());
        dev.write(QUEUE_NOTIFY, 0);
        assert!(dev.needs_processing());
    }

    #[test]
    fn test_vsock_config_generation() {
        let dev = VirtioVsock::new();
        assert_eq!(dev.read(CONFIG_GENERATION), 0);
    }

    #[test]
    fn test_vsock_handle_request_sends_rst() {
        let mut dev = VirtioVsock::new();
        // Simulate a guest connection request
        let hdr = VsockHeader {
            src_cid: GUEST_CID,
            dst_cid: 2, // Host
            src_port: 1234,
            dst_port: 80,
            len: 0,
            r#type: 1,
            op: VsockOp::Request,
            flags: 0,
            buf_alloc: 4096,
            fwd_cnt: 0,
        };
        dev.tx_packets.push((hdr, Vec::new()));
        dev.handle_tx_packets();
        // Should have queued a RST response
        assert_eq!(dev.rx_queue.len(), 1);
        let (resp, _) = &dev.rx_queue[0];
        assert_eq!(resp.op, VsockOp::Rst);
        assert_eq!(resp.src_cid, 2); // From host
        assert_eq!(resp.dst_cid, GUEST_CID);
        assert_eq!(resp.src_port, 80);
        assert_eq!(resp.dst_port, 1234);
    }

    #[test]
    fn test_vsock_handle_credit_request() {
        let mut dev = VirtioVsock::new();
        let hdr = VsockHeader {
            src_cid: GUEST_CID,
            dst_cid: 2,
            src_port: 1234,
            dst_port: 80,
            len: 0,
            r#type: 1,
            op: VsockOp::CreditRequest,
            flags: 0,
            buf_alloc: 0,
            fwd_cnt: 0,
        };
        dev.tx_packets.push((hdr, Vec::new()));
        dev.handle_tx_packets();
        assert_eq!(dev.rx_queue.len(), 1);
        let (resp, _) = &dev.rx_queue[0];
        assert_eq!(resp.op, VsockOp::CreditUpdate);
        assert_eq!(resp.buf_alloc, 65536);
    }

    #[test]
    fn test_vsock_handle_shutdown() {
        let mut dev = VirtioVsock::new();
        // Add a connection
        dev.connections.push(VsockConnection {
            src_cid: GUEST_CID,
            src_port: 1234,
            dst_cid: 2,
            dst_port: 80,
        });
        // Send shutdown
        let hdr = VsockHeader {
            src_cid: GUEST_CID,
            dst_cid: 2,
            src_port: 1234,
            dst_port: 80,
            len: 0,
            r#type: 1,
            op: VsockOp::Shutdown,
            flags: 0,
            buf_alloc: 0,
            fwd_cnt: 0,
        };
        dev.tx_packets.push((hdr, Vec::new()));
        dev.handle_tx_packets();
        assert!(dev.connections.is_empty());
        assert_eq!(dev.rx_queue.len(), 1);
        assert_eq!(dev.rx_queue[0].0.op, VsockOp::Rst);
    }

    #[test]
    fn test_vsock_handle_rw_silently_consumed() {
        let mut dev = VirtioVsock::new();
        let hdr = VsockHeader {
            src_cid: GUEST_CID,
            dst_cid: 2,
            src_port: 1234,
            dst_port: 80,
            len: 5,
            r#type: 1,
            op: VsockOp::Rw,
            flags: 0,
            buf_alloc: 4096,
            fwd_cnt: 0,
        };
        dev.tx_packets.push((hdr, b"hello".to_vec()));
        dev.handle_tx_packets();
        // No response for data packets without backend
        assert!(dev.rx_queue.is_empty());
    }
}
