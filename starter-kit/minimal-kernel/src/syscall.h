#ifndef SYSCALL_H
#define SYSCALL_H

#include <stdint.h>

// Syscall numbers (in a7 register)
#define SYS_WRITE   1
#define SYS_EXIT    2
#define SYS_YIELD   3
#define SYS_GETPID  4

void syscall_dispatch(void *frame);

#endif
