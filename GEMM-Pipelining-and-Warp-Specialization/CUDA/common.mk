# Shared build rules for the CUDA benchmarks.
# A benchmark Makefile sets CUFILES (one or more .cu files, each a standalone
# program) and includes this file. Every variant is then built with identical
# flags so their reported timings are comparable.

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_80

EXECUTABLES := $(CUFILES:.cu=.exe)

# Each .cu #includes the benchmark header and ../../common/polybench.c
# directly, so rebuild when any of those change.
COMMON_DEPS := $(wildcard *.cuh) $(wildcard ../../common/*.h) $(wildcard ../../common/*.c)

.PHONY: all clean

all: $(EXECUTABLES)

%.exe: %.cu $(COMMON_DEPS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f *~ *.exe
