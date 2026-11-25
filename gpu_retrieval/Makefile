CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -march=native
NVCC     ?= nvcc
# Set to your GPU's compute capability (T4=75, A100=80, RTX30xx=86, RTX40xx=89)
ARCH     ?= native
NVFLAGS  ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp -lineinfo

all: cpu

cpu: bench_cpu
bench_cpu: src/main_cpu.cpp src/cpu_pipeline.cpp src/cpu_pipeline.h src/common.h
	$(CXX) $(CXXFLAGS) src/main_cpu.cpp src/cpu_pipeline.cpp -o $@

gpu: bench_gpu
bench_gpu: src/main_gpu.cu src/gpu_pipeline.cu src/gpu_pipeline.h src/cpu_pipeline.cpp src/cpu_pipeline.h src/common.h
	$(NVCC) $(NVFLAGS) src/main_gpu.cu src/gpu_pipeline.cu src/cpu_pipeline.cpp -o $@

clean:
	rm -f bench_cpu bench_gpu
