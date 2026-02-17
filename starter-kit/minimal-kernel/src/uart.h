#ifndef UART_H
#define UART_H

#include <stdint.h>
#include <stdarg.h>

// UART base address (16550 compatible, matches microvm memory map)
#define UART_BASE 0x10000000UL

// UART registers
#define UART_THR  0  // Transmit Holding Register
#define UART_RBR  0  // Receive Buffer Register
#define UART_IER  1  // Interrupt Enable Register
#define UART_LSR  5  // Line Status Register
#define UART_LSR_TX_EMPTY 0x20

void uart_init(void);
void uart_putchar(char c);
void uart_puts(const char *s);
void uart_printf(const char *fmt, ...);
char uart_getchar(void);

#endif
