// syscall.c — 4 Syscalls: write, exit, yield, getpid

#include "syscall.h"
#include "uart.h"
#include "sched.h"

// The trap frame layout matches our register save in boot.S
// a7 is at offset 88, a0 at 32, a1 at 40, a2 at 48
typedef struct {
    uint64_t ra, t0, t1, t2;
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
    uint64_t t3, t4, t5, t6;
    uint64_t s0, s1;
} trap_frame_t;

void syscall_dispatch(void *frame) {
    trap_frame_t *f = (trap_frame_t *)frame;
    uint64_t syscall_num = f->a7;

    switch (syscall_num) {
    case SYS_WRITE: {
        // write(fd, buf, len) — fd ignored, writes to UART
        const char *buf = (const char *)f->a1;
        uint64_t len = f->a2;
        for (uint64_t i = 0; i < len; i++)
            uart_putchar(buf[i]);
        f->a0 = len;  // Return bytes written
        break;
    }
    case SYS_EXIT:
        uart_printf("[syscall] task %d called exit(%d)\n",
                     current_pid(), (int)f->a0);
        // Mark task dead and yield
        yield();
        break;
    case SYS_YIELD:
        yield();
        break;
    case SYS_GETPID:
        f->a0 = (uint64_t)current_pid();
        break;
    default:
        uart_printf("[syscall] unknown syscall %d from task %d\n",
                     (int)syscall_num, current_pid());
        f->a0 = (uint64_t)-1;
        break;
    }
}
