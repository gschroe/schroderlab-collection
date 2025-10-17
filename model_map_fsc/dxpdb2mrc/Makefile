# Compiler and flags
CC      = gcc
CFLAGS  = -O2 -Wall -std=c99
LDFLAGS = -L/opt/homebrew/lib -lm -lfftw3f

# Source files and corresponding targets
TARGETS = dxpdb2mrc dxpdb2mrc-bfactor
SOURCES = dxpdb2mrc.c dxpdb2mrc-bfactor.c

.PHONY: all clean

all: $(TARGETS)

dxpdb2mrc: dxpdb2mrc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

dxpdb2mrc-bfactor: dxpdb2mrc-bfactor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)


