# CUDA compiler
NVCC = nvcc

# Flags for the CUDA compiler
NVCC_FLAGS = -std=c++11 -O3

# Include directories
INCLUDES = -I../include

# Libraries to link
LIBS = -lcudart

# Source files
SRC = main.cu GSZ_entry.cu GSZ.cu GSZ_timer.cu
# Output executable
TARGET = main

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
