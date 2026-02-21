#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/reboot.h>
#include <termios.h>

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

/* Write via file descriptor (uses kernel tty layer) */
static void fd_puts(int fd, const char *s) {
    if (fd >= 0)
        write(fd, s, strlen(s));
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

    uart_puts("\n");
    uart_puts("  ========================================\n");
    uart_puts("  |  Linux booted on microvm!            |\n");
    uart_puts("  |  Hello from userspace init!          |\n");
    uart_puts("  ========================================\n");
    uart_puts("\n");

    /* Test 1: tty layer via /dev/console */
    uart_puts("[test] Testing tty write via /dev/console... ");
    int con_fd = open("/dev/console", O_WRONLY | O_NOCTTY);
    if (con_fd >= 0) {
        const char *msg = "[tty:console] Hello from tty layer!\n";
        ssize_t n = write(con_fd, msg, strlen(msg));
        if (n > 0) {
            uart_puts("OK (wrote via /dev/console)\n");
        } else {
            uart_puts("FAIL (write returned <= 0)\n");
        }
        close(con_fd);
    } else {
        uart_puts("FAIL (open failed)\n");
    }

    /* Test 2: tty layer via /dev/ttyS0 */
    uart_puts("[test] Testing tty write via /dev/ttyS0... ");
    int tty_fd = open("/dev/ttyS0", O_WRONLY | O_NOCTTY);
    if (tty_fd >= 0) {
        const char *msg = "[tty:ttyS0] Hello from tty layer!\n";
        ssize_t n = write(tty_fd, msg, strlen(msg));
        if (n > 0) {
            uart_puts("OK (wrote via /dev/ttyS0)\n");
        } else {
            uart_puts("FAIL (write returned <= 0)\n");
        }
        close(tty_fd);
    } else {
        uart_puts("FAIL (open failed)\n");
    }

    /* Test 3: printf (stdout → console) */
    uart_puts("[test] Testing printf (stdout)... ");
    /* Redirect stdout to /dev/console */
    int std_fd = open("/dev/console", O_WRONLY | O_NOCTTY);
    if (std_fd >= 0) {
        dup2(std_fd, STDOUT_FILENO);
        dup2(std_fd, STDERR_FILENO);
        close(std_fd);
    }
    printf("[printf] Hello from printf!\n");
    fflush(stdout);
    uart_puts("(check above for printf output)\n");

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

    /* Show interrupt counts */
    f = fopen("/proc/interrupts", "r");
    if (f) {
        uart_puts("[init] Interrupts:\n");
        while (fgets(buf, sizeof(buf), f)) {
            uart_puts("  ");
            uart_puts(buf);
        }
        fclose(f);
    }

    uart_puts("[init] All done. Powering off.\n");
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 0;
}
