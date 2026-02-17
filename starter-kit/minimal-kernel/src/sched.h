#ifndef SCHED_H
#define SCHED_H

#include <stdint.h>

#define MAX_TASKS    4
#define STACK_SIZE   4096

typedef enum { TASK_READY, TASK_RUNNING, TASK_DEAD } task_state_t;

typedef struct {
    uint64_t ra, sp;
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
} context_t;

typedef struct {
    int           pid;
    task_state_t  state;
    context_t     ctx;
    uint8_t       stack[STACK_SIZE] __attribute__((aligned(16)));
} task_t;

void sched_init(void);
void task_create(void (*entry)(void), const char *name);
void schedule(void);
void yield(void);
int  current_pid(void);

#endif
