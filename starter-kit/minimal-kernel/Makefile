# Makefile — Cross-compile MyOS for RISC-V

CROSS   = riscv64-unknown-elf-
CC      = $(CROSS)gcc
AS      = $(CROSS)gcc
LD      = $(CROSS)ld
OBJCOPY = $(CROSS)objcopy

# Try riscv64-elf- prefix if riscv64-unknown-elf- not found
ifeq ($(shell which $(CC) 2>/dev/null),)
    CROSS = riscv64-elf-
    CC    = $(CROSS)gcc
    AS    = $(CROSS)gcc
    LD    = $(CROSS)ld
    OBJCOPY = $(CROSS)objcopy
endif

CFLAGS  = -Wall -Wextra -O2 -ffreestanding -nostdlib -nostartfiles \
          -march=rv64gc -mabi=lp64d -mcmodel=medany -Isrc
ASFLAGS = -march=rv64gc -mabi=lp64d -mcmodel=medany
LDFLAGS = -T linker.ld -nostdlib

SRC_C = src/main.c src/uart.c src/trap.c src/mm.c src/sched.c src/syscall.c
SRC_S = src/boot.S
OBJ   = $(SRC_S:.S=.o) $(SRC_C:.c=.o)

.PHONY: all clean run

all: kernel.bin
	@echo ""
	@echo "  ✅ kernel.bin built successfully!"
	@echo "  Run with: microvm run --kernel kernel.bin --load-addr 0x80000000"
	@echo ""

kernel.elf: $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $^

kernel.bin: kernel.elf
	$(OBJCOPY) -O binary $< $@

src/%.o: src/%.S
	$(AS) $(ASFLAGS) -c -o $@ $<

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: kernel.bin
	microvm run --kernel kernel.bin --load-addr 0x80000000

clean:
	rm -f src/*.o kernel.elf kernel.bin
