def:
	make -C mamba3_mlx self-guided PROMPT="Who are you?"

# Forward sjd targets to inner Makefile so `make sjd` works from repo root.
# All variables are passed through (PROMPT, MODE, MAX_TOK, STREAM, K, …).
sjd sjd-self sjd-math sjd-daily sjd-emotion sjd-email sjd-movie sjd-deep sjd-syscall:
	$(MAKE) -C mamba3_mlx $@ $(MAKEOVERRIDES)
