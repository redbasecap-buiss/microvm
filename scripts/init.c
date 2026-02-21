#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/mman.h>
#include <linux/reboot.h>

/* Direct UART MMIO output — bypasses tty layer for reliable output */
static volatile unsigned char *uart_base = NULL;

static void uart_putc(char c) {
    if (uart_base) {
        while (!(uart_base[5] & 0x20)) {} /* Wait for THRE */
        uart_base[0] = c;
    }
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

int main(void) {
    mount("proc", "/proc", "proc", 0, NULL);
    mount("sysfs", "/sys", "sysfs", 0, NULL);
    mount("devtmpfs", "/dev", "devtmpfs", 0, NULL);

    /* Map UART MMIO for direct output */
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd >= 0) {
        uart_base = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED,
                         mem_fd, 0x10000000);
        if (uart_base == MAP_FAILED)
            uart_base = NULL;
        close(mem_fd);
    }

    /* Fallback: try /dev/ttyS0 */
    if (!uart_base) {
        int fd = open("/dev/ttyS0", O_WRONLY);
        if (fd >= 0) {
            dup2(fd, 1);
            if (fd > 1) close(fd);
        }
    }

    uart_puts("\n");
    uart_puts("  ========================================\n");
    uart_puts("  |  Linux booted on microvm!            |\n");
    uart_puts("  |  Hello from userspace init!          |\n");
    uart_puts("  ========================================\n");
    uart_puts("\n");

    FILE *f;
    char buf[256];

    f = fopen("/proc/version", "r");
    if (f) {
        uart_puts("[init] ");
        if (fgets(buf, sizeof(buf), f))
            uart_puts(buf);
        fclose(f);
    }

    f = fopen("/proc/cpuinfo", "r");
    if (f) {
        uart_puts("[init] CPU info:\n");
        while (fgets(buf, sizeof(buf), f)) {
            if (buf[0] == '\n') break;
            uart_puts("  ");
            uart_puts(buf);
        }
        fclose(f);
    }

    f = fopen("/proc/meminfo", "r");
    if (f) {
        int lines = 0;
        while (fgets(buf, sizeof(buf), f) && lines < 3) {
            uart_puts("[init] ");
            uart_puts(buf);
            lines++;
        }
        fclose(f);
    }

    f = fopen("/proc/uptime", "r");
    if (f) {
        uart_puts("[init] Uptime: ");
        if (fgets(buf, sizeof(buf), f))
            uart_puts(buf);
        fclose(f);
    }

    uart_puts("[init] All done. Powering off.\n");
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 0;
}
