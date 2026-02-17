# Makefile — Rust RISC-V Kernel

OBJCOPY = riscv64-unknown-elf-objcopy
ifeq ($(shell which $(OBJCOPY) 2>/dev/null),)
    OBJCOPY = riscv64-elf-objcopy
endif
ifeq ($(shell which $(OBJCOPY) 2>/dev/null),)
    OBJCOPY = llvm-objcopy
endif

.PHONY: all clean run

all: kernel.bin
	@echo ""
	@echo "  ✅ Rust kernel.bin built!"
	@echo "  Run with: microvm run --kernel kernel.bin --load-addr 0x80000000"
	@echo ""

kernel.bin: target/riscv64gc-unknown-none-elf/release/rust-kernel
	$(OBJCOPY) -O binary $< $@

target/riscv64gc-unknown-none-elf/release/rust-kernel: src/*.rs src/boot.S Cargo.toml linker.ld
	cargo build --release

run: kernel.bin
	microvm run --kernel kernel.bin --load-addr 0x80000000

clean:
	cargo clean
	rm -f kernel.bin
