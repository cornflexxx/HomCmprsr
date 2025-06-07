NVCC = nvcc -ccbin mpic++

NVCC_FLAGS = -std=c++11

INCLUDES = -I../include

LIBS = -lcudart

SRC = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_hom.cu 
TARGET = main_hom

SRC_2 = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_allred.cu AWcudaAllreduce.cu
TARGET_2 = main_allred

SRC_3 = GSZ_entry.cu GSZ_timer.cu comprs_test.cu GSZ.cu  main_naw_allred.cu NonAWcudaAllreduce.cu
TARGET_3 = naw_allred

compression: $(TARGET)

allreduce: $(TARGET_2)

naw_allreduce: $(TARGET_3)

$(TARGET_2): $(SRC_2)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_2) -o $(TARGET_2) $(LIBS)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LIBS)

$(TARGET_3): $(SRC_3)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_3) -o $(TARGET_3) $(LIBS)

clean:
	rm -f $(TARGET) $(TARGET_2) $(TARGET_3)
