#===============================================================================
# User Options
#===============================================================================

COMPILER    = nvcc
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
TABLE       = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = kernel

source = \
kernel.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Regular gcc Compiler
ifeq ($(COMPILER),nvcc)
  CC = nvcc
endif

# Standard Flags
CFLAGS := -arch sm_70 -I/usr/local/cuda/include

# Linker Flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3 --use_fast_math
endif

# Exponential Table Build
ifeq ($(TABLE), yes)
	CFLAGS += -DTABLE
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source)

run:
	./$(program)
