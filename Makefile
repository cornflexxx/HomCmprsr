NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3
INCLUDES = -I../include
LIBS = -lcudart

SRC = main.cu GSZ_entry.cu GSZ.cu GSZ_timer.cu
TARGET = main

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
