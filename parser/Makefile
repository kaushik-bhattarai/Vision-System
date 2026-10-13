# Build:  CUDA_VER=13.2 make 
ifeq ($(CUDA_VER),)
$(error CUDA_VER is not set, e.g. CUDA_VER=13.2 make)
endif
DS_PATH ?= /opt/nvidia/deepstream/deepstream
CXX     ?= g++
CFLAGS  := -Wall -std=c++14 -shared -fPIC -O2 -Wno-deprecated-declarations \
           -I$(DS_PATH)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
LIBS    := -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lnvinfer

libnvdsparse_yolo26.so: nvdsparse_yolo26.cpp
	$(CXX) -o $@ $(CFLAGS) $< $(LIBS)

clean:
	rm -f libnvdsparse_yolo26.so
