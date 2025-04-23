# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -O2 -std=c++11
NVFLAGS = -O2 -arch=sm_60  # Adjust for your GPU arch if needed

# Source files
CPU_SRC = cpu_prefix_sum.cpp
NAIVE_GPU_SRC = gpu_naive_scan.cu
BLELLOCH_SRC = blelloch_scan.cu
COMPACTION_SRC = stream_compaction.cu
HISTOGRAM_SRC = histogram.cu
MAIN_SRC = main.cpp

# Output binary
TARGET = prefix_sum_benchmark

# Object files
OBJS = $(CPU_SRC:.cpp=.o) \
       $(NAIVE_GPU_SRC:.cu=.o) \
       $(BLELLOCH_SRC:.cu=.o) \
       $(COMPACTION_SRC:.cu=.o) \
       $(HISTOGRAM_SRC:.cu=.o) \
       $(MAIN_SRC:.cpp=.o)

# Build rules
all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) $(OBJS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f *.o $(TARGET)
