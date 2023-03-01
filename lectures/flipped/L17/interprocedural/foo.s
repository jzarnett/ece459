	.text
	.intel_syntax noprefix
	.file	"foo.0db40bb9-cgu.0"
	.section	.text.flag,"ax",@progbits
	.globl	flag
	.p2align	4, 0x90
	.type	flag,@function
flag:
	.cfi_startproc
	mov	al, 1
	ret
.Lfunc_end0:
	.size	flag, .Lfunc_end0-flag
	.cfi_endproc

	.section	".note.GNU-stack","",@progbits
