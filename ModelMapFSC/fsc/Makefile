# Compiler and flags
CC      = gcc-12
CFLAGS  = -O2 -Wall -std=c99 -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lm -lfftw3

# Target executable
TARGET  = dxfsc
SRC     = dxfsc.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

