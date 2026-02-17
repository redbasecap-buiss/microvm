# Tutorial 05: Scheduler — Round-Robin Multitasking

> ⏱️ Time: 15 minutes

An OS isn't an OS without multitasking. We'll implement a simple round-robin scheduler that switches between tasks on every timer tick.

## Concepts

**Cooperative scheduling**: Tasks call `yield()` to give up the CPU.
**Preemptive scheduling**: The timer interrupt forces a context switch.

Our scheduler does both — tasks can yield voluntarily, and the timer ensures no task hogs the CPU.

## Step 1: Task Structure

Each task needs saved registers and a stack:

```c
typedef struct {
    uint64_t ra, sp;
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
} context_t;

typedef struct {
    int           pid;
    task_state_t  state;   // READY, RUNNING, DEAD
    context_t     ctx;
    uint8_t       stack[4096] __attribute__((aligned(16)));
} task_t;
```

We only save **callee-saved registers** (s0-s11, ra, sp). The C calling convention guarantees the caller saves everything else.

## Step 2: Context Switch (Assembly)

The heart of the scheduler — swap one task's registers for another's:

```asm
context_switch:
    sd ra,  0(a0)      # Save old task's registers
    sd sp,  8(a0)
    sd s0, 16(a0)
    # ... save s1-s11 ...

    ld ra,  0(a1)      # Load new task's registers
    ld sp,  8(a1)
    ld s0, 16(a1)
    # ... load s1-s11 ...
    ret                  # Jump to new task's ra
```

**Why `ret`?** After loading the new task's `ra`, `ret` jumps to wherever that task was last saved — either back into its code, or to its entry point (for new tasks).

## Step 3: Task Creation

```c
void task_create(void (*entry)(void), const char *name) {
    task_t *t = &tasks[num_tasks];
    t->pid = num_tasks + 1;
    t->state = TASK_READY;

    uint64_t stack_top = (uint64_t)&t->stack[STACK_SIZE] & ~0xF;
    t->ctx.sp = stack_top;
    t->ctx.ra = (uint64_t)entry;  // First context_switch will "ret" here!

    num_tasks++;
}
```

**The trick**: Set `ra` to the entry function. When `context_switch` does `ret`, it jumps right into the task!

## Step 4: Round-Robin Scheduling

```c
void schedule(void) {
    int prev = current_task;

    // Find next ready task
    for (int i = 1; i <= num_tasks; i++) {
        int idx = (current_task + i) % num_tasks;
        if (tasks[idx].state != TASK_DEAD) {
            tasks[prev].state = TASK_READY;
            current_task = idx;
            tasks[idx].state = TASK_RUNNING;
            context_switch(&tasks[prev].ctx, &tasks[idx].ctx);
            return;
        }
    }

    // No tasks left
    while (1) asm volatile("wfi");
}
```

## Step 5: Preemptive Switching

In the timer interrupt handler, just call `schedule()`:

```c
if (code == CAUSE_MACHINE_TIMER) {
    tick_count++;
    timer_set_next();
    schedule();  // Force context switch!
}
```

Now tasks get preempted automatically — even if they don't call `yield()`.

## Execution Flow

```
Timer fires → trap_entry → save regs → trap_handler
  → schedule() → find next task → context_switch
    → save old sp/ra/s0-s11
    → load new sp/ra/s0-s11
    → ret → new task continues
```

## Run It

With the starter kit:
```bash
cd starter-kit/minimal-kernel
make && make run
```

You'll see Task A and Task B alternating:
```
[Task A] Hello! tick=0, pid=1, iteration=0
[Task B] World! tick=1, pid=2, iteration=0
[Task A] Hello! tick=2, pid=1, iteration=1
...
```

## What's Next?

→ [Tutorial 06: Rust Kernel](06-rust-kernel.md) — Rewrite it all in Rust!
