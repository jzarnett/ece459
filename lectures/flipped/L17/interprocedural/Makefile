EMIT=--emit asm -Cllvm-args=--x86-asm-syntax=intel
OPT=-C overflow-checks=n -C opt-level=3

# on ecetesla
all:
	# build libfoo
	rustc $(EMIT) $(OPT) --crate-type=rlib foo.rs
	rustc $(OPT) --crate-type=rlib foo.rs
	# build main
	rustc $(EMIT) $(OPT) --extern foo=libfoo.rlib main.rs
	# rustc $(OPT) --extern foo=libfoo.rlib main.rs

.PHONY: clean
clean:
	rm -f *.rcgu.o *.rlib *.cdylib *.o *.s main
