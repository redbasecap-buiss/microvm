// uart.c — 16550 UART Driver (putchar, puts, printf-lite)

#include "uart.h"

static volatile uint8_t *const uart = (volatile uint8_t *)UART_BASE;

void uart_init(void) {
    // Disable interrupts
    uart[UART_IER] = 0x00;
    // Enable FIFO
    uart[2] = 0x07;  // FCR: enable & clear FIFOs
    // 8 bits, no parity, 1 stop bit
    uart[3] = 0x03;  // LCR
    // Enable receive interrupts
    uart[UART_IER] = 0x01;
}

void uart_putchar(char c) {
    while ((uart[UART_LSR] & UART_LSR_TX_EMPTY) == 0)
        ;
    uart[UART_THR] = (uint8_t)c;
}

char uart_getchar(void) {
    while ((uart[UART_LSR] & 0x01) == 0)
        ;
    return (char)uart[UART_RBR];
}

void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n')
            uart_putchar('\r');
        uart_putchar(*s++);
    }
}

// Minimal printf: supports %s, %d, %x, %p, %c, %%
static void print_int(int64_t val, int base, int is_signed) {
    char buf[24];
    int i = 0;
    uint64_t uval;

    if (is_signed && val < 0) {
        uart_putchar('-');
        uval = (uint64_t)(-val);
    } else {
        uval = (uint64_t)val;
    }

    if (uval == 0) {
        uart_putchar('0');
        return;
    }

    while (uval > 0) {
        int d = uval % base;
        buf[i++] = d < 10 ? '0' + d : 'a' + d - 10;
        uval /= base;
    }

    while (--i >= 0)
        uart_putchar(buf[i]);
}

void uart_printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);

    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            if (*fmt == '\n') uart_putchar('\r');
            uart_putchar(*fmt);
            continue;
        }
        fmt++;
        switch (*fmt) {
        case 's': uart_puts(va_arg(ap, const char *)); break;
        case 'd': print_int(va_arg(ap, int), 10, 1); break;
        case 'x': print_int(va_arg(ap, unsigned int), 16, 0); break;
        case 'p': uart_puts("0x"); print_int(va_arg(ap, uint64_t), 16, 0); break;
        case 'c': uart_putchar((char)va_arg(ap, int)); break;
        case '%': uart_putchar('%'); break;
        default:  uart_putchar('%'); uart_putchar(*fmt); break;
        }
    }

    va_end(ap);
}
