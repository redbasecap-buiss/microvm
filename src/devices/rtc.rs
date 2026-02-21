/// Goldfish RTC (Real-Time Clock) device
///
/// Provides wall-clock time to the guest via a simple MMIO interface.
/// Linux has built-in support for "google,goldfish-rtc".
///
/// Register map (all 32-bit reads):
///   0x00  TIME_LOW        — low 32 bits of nanoseconds since epoch
///   0x04  TIME_HIGH       — high 32 bits of nanoseconds since epoch
///   0x08  ALARM_LOW       — alarm low (ns)
///   0x0C  ALARM_HIGH      — alarm high (ns)
///   0x10  IRQ_ENABLED     — alarm IRQ enable (1=enabled)
///   0x14  CLEAR_ALARM     — clear alarm (write any value)
///   0x18  ALARM_STATUS    — alarm status (1=alarm fired)
///   0x1C  CLEAR_INTERRUPT — clear interrupt (write any value)
use std::time::{SystemTime, UNIX_EPOCH};

pub struct GoldfishRtc {
    /// Cached time in nanoseconds (updated on TIME_LOW read, latched for TIME_HIGH)
    latched_ns: u64,
    /// Alarm target time in nanoseconds
    alarm_ns: u64,
    /// Whether alarm IRQ is enabled
    irq_enabled: bool,
    /// Whether alarm has fired (pending interrupt)
    alarm_fired: bool,
}

impl Default for GoldfishRtc {
    fn default() -> Self {
        Self::new()
    }
}

impl GoldfishRtc {
    pub fn new() -> Self {
        Self {
            latched_ns: 0,
            alarm_ns: 0,
            irq_enabled: false,
            alarm_fired: false,
        }
    }

    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Check if RTC has a pending interrupt (alarm fired and IRQ enabled).
    pub fn has_interrupt(&self) -> bool {
        self.irq_enabled && self.alarm_fired
    }

    /// Tick: check if alarm has fired (call from VM loop).
    pub fn tick(&mut self) {
        if self.irq_enabled && !self.alarm_fired && self.alarm_ns > 0 {
            let now = Self::now_ns();
            if now >= self.alarm_ns {
                self.alarm_fired = true;
            }
        }
    }

    /// Read a 32-bit register at the given offset.
    /// Reading TIME_LOW latches the full 64-bit time so TIME_HIGH is consistent.
    pub fn read(&mut self, offset: u64) -> u32 {
        match offset {
            0x00 => {
                // TIME_LOW — latch current time
                self.latched_ns = Self::now_ns();
                self.latched_ns as u32
            }
            0x04 => {
                // TIME_HIGH — return upper 32 bits of latched time
                (self.latched_ns >> 32) as u32
            }
            0x08 => {
                // ALARM_LOW
                self.alarm_ns as u32
            }
            0x0C => {
                // ALARM_HIGH
                (self.alarm_ns >> 32) as u32
            }
            0x10 => {
                // IRQ_ENABLED
                self.irq_enabled as u32
            }
            0x18 => {
                // ALARM_STATUS
                self.alarm_fired as u32
            }
            _ => 0,
        }
    }

    /// Write a 32-bit register.
    pub fn write(&mut self, offset: u64, val: u64) {
        let val32 = val as u32;
        match offset {
            0x08 => {
                // ALARM_LOW
                self.alarm_ns = (self.alarm_ns & 0xFFFF_FFFF_0000_0000) | val32 as u64;
            }
            0x0C => {
                // ALARM_HIGH
                self.alarm_ns = (self.alarm_ns & 0x0000_0000_FFFF_FFFF) | ((val32 as u64) << 32);
            }
            0x10 => {
                // IRQ_ENABLED
                self.irq_enabled = val32 != 0;
            }
            0x14 => {
                // CLEAR_ALARM
                self.alarm_ns = 0;
                self.alarm_fired = false;
            }
            0x1C => {
                // CLEAR_INTERRUPT
                self.alarm_fired = false;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtc_returns_nonzero_time() {
        let mut rtc = GoldfishRtc::new();
        let low = rtc.read(0x00);
        let high = rtc.read(0x04);
        let ns = ((high as u64) << 32) | (low as u64);
        // Should be after year 2020 (1577836800 seconds = 0x5E0BE100)
        assert!(ns > 1_577_836_800_000_000_000);
    }

    #[test]
    fn test_rtc_latching() {
        let mut rtc = GoldfishRtc::new();
        // Read TIME_LOW to latch
        let low = rtc.read(0x00);
        let high = rtc.read(0x04);
        // Read TIME_HIGH again — should return same latched value
        let high2 = rtc.read(0x04);
        assert_eq!(high, high2);
        // Sanity: low was captured
        assert_eq!(low, rtc.latched_ns as u32);
    }

    #[test]
    fn test_rtc_alarm_status_zero() {
        let mut rtc = GoldfishRtc::new();
        assert_eq!(rtc.read(0x18), 0);
    }

    #[test]
    fn test_rtc_write_alarm() {
        let mut rtc = GoldfishRtc::new();
        rtc.write(0x08, 0xDEADBEEF); // ALARM_LOW
        rtc.write(0x0C, 0x12345678); // ALARM_HIGH
        assert_eq!(rtc.read(0x08), 0xDEADBEEF);
        assert_eq!(rtc.read(0x0C), 0x12345678);
        assert_eq!(rtc.alarm_ns, 0x12345678_DEADBEEF);
    }

    #[test]
    fn test_rtc_no_interrupt_by_default() {
        let rtc = GoldfishRtc::new();
        assert!(!rtc.has_interrupt());
    }

    #[test]
    fn test_rtc_alarm_fires() {
        let mut rtc = GoldfishRtc::new();
        // Set alarm to 1 ns (already in the past)
        rtc.write(0x08, 1); // ALARM_LOW = 1
        rtc.write(0x0C, 0); // ALARM_HIGH = 0
        rtc.write(0x10, 1); // IRQ_ENABLED = true
        assert!(!rtc.has_interrupt());
        rtc.tick();
        assert!(rtc.has_interrupt());
        assert_eq!(rtc.read(0x18), 1); // ALARM_STATUS = fired
    }

    #[test]
    fn test_rtc_clear_interrupt() {
        let mut rtc = GoldfishRtc::new();
        rtc.write(0x08, 1);
        rtc.write(0x10, 1);
        rtc.tick();
        assert!(rtc.has_interrupt());
        rtc.write(0x1C, 1); // CLEAR_INTERRUPT
        assert!(!rtc.has_interrupt());
    }

    #[test]
    fn test_rtc_clear_alarm() {
        let mut rtc = GoldfishRtc::new();
        rtc.write(0x08, 1);
        rtc.write(0x10, 1);
        rtc.tick();
        assert!(rtc.has_interrupt());
        rtc.write(0x14, 1); // CLEAR_ALARM
        assert!(!rtc.has_interrupt());
        assert_eq!(rtc.alarm_ns, 0);
    }

    #[test]
    fn test_rtc_irq_disabled_no_fire() {
        let mut rtc = GoldfishRtc::new();
        rtc.write(0x08, 1); // alarm in the past
                            // IRQ not enabled
        rtc.tick();
        assert!(!rtc.has_interrupt());
        assert!(!rtc.alarm_fired); // tick doesn't fire if irq disabled
    }

    #[test]
    fn test_rtc_irq_enable_read() {
        let mut rtc = GoldfishRtc::new();
        assert_eq!(rtc.read(0x10), 0);
        rtc.write(0x10, 1);
        assert_eq!(rtc.read(0x10), 1);
    }
}
