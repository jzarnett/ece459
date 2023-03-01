	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0
	.globl	_flag
	.p2align	2
_flag:
	.cfi_startproc
	mov	w0, #1
	ret
	.cfi_endproc

.subsections_via_symbols
