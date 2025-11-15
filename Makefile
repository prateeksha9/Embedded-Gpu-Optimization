# Makefile for compiling CUDA programs

NVCC = nvcc
INCLUDE = -I ./
SRC1 = chaerina_shresth4_prateekr_1.cu
SRC2 = chaerina_shresth4_prateekr_2.cu
SRC3 = chaerina_shresth4_prateekr_3.cu
OUT1 = prob_1
OUT2 = prob_2
OUT3 = prob_3

# Default target: build all
all: $(OUT1) $(OUT2) $(OUT3)

$(OUT1): $(SRC1)
	$(NVCC) $(SRC1) -o $(OUT1) $(INCLUDE)

$(OUT2): $(SRC2)
	$(NVCC) $(SRC2) -o $(OUT2) $(INCLUDE)

$(OUT3): $(SRC3)
	$(NVCC) $(SRC3) -o $(OUT3) $(INCLUDE)

# Cleanup
clean:
	rm -f $(OUT1) $(OUT2) $(OUT3)