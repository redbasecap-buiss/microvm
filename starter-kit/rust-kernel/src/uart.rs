/// 16550 UART Driver

pub struct Uart {
    base: usize,
}

impl Uart {
    pub const fn new(base: usize) -> Self {
        Uart { base }
    }

    fn reg(&self, offset: usize) -> *mut u8 {
        (self.base + offset) as *mut u8
    }

    pub fn init(&self) {
        unsafe {
            self.reg(1).write_volatile(0x00); // Disable interrupts
            self.reg(2).write_volatile(0x07); // Enable FIFO
            self.reg(3).write_volatile(0x03); // 8N1
            self.reg(1).write_volatile(0x01); // Enable RX interrupt
        }
    }

    pub fn putchar(&self, c: u8) {
        unsafe {
            // Wait for TX empty
            while self.reg(5).read_volatile() & 0x20 == 0 {}
            self.reg(0).write_volatile(c);
        }
    }

    pub fn puts(&self, s: &str) {
        for b in s.bytes() {
            if b == b'\n' {
                self.putchar(b'\r');
            }
            self.putchar(b);
        }
    }

    pub fn put_dec(&self, mut val: u64) {
        if val == 0 {
            self.putchar(b'0');
            return;
        }
        let mut buf = [0u8; 20];
        let mut i = 0;
        while val > 0 {
            buf[i] = b'0' + (val % 10) as u8;
            val /= 10;
            i += 1;
        }
        while i > 0 {
            i -= 1;
            self.putchar(buf[i]);
        }
    }

    pub fn put_hex(&self, mut val: u64) {
        self.puts("0x");
        if val == 0 {
            self.putchar(b'0');
            return;
        }
        let mut buf = [0u8; 16];
        let mut i = 0;
        while val > 0 {
            let d = (val & 0xF) as u8;
            buf[i] = if d < 10 { b'0' + d } else { b'a' + d - 10 };
            val >>= 4;
            i += 1;
        }
        while i > 0 {
            i -= 1;
            self.putchar(buf[i]);
        }
    }
}
