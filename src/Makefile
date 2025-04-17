NVCC = nvcc

NVCC_FLAGS = -std=c++11

INCLUDES = -I../include

LIBS = -lcudart

SRC = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_hom.cu
TARGET = main_hom


all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
