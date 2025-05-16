OPSYS=$(shell uname)

PLATFORM=$(shell uname -p)
ARCH=.$(PLATFORM)

CXX:=ccache $(CXX)

CXXFLAGS+=-std=c++17

CPPFLAGS+=$(LOCAL_CFLAGS)
LDFLAGS+=$(LOCAL_LDFLAGS)

# Find LibTorch path - use the system installation path
LIBTORCH_PATH?=/usr/lib/libtorch

# Include paths for all dependencies
CPPFLAGS+=-I./include -I/usr/include/eigen3 -I/usr/include/jsoncpp -I/usr/include/boost
CPPFLAGS+=-I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include

# Library paths and linking
LDFLAGS+=-lyaml-cpp -ljsoncpp -lboost_system -lboost_iostreams
LDFLAGS+=-L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10

# Add runtime path to ensure libraries are found
LDFLAGS+=-Wl,-rpath,$(LIBTORCH_PATH)/lib

# ABI compatibility flag for LibTorch
LDFLAGS+=-D_GLIBCXX_USE_CXX11_ABI=1

# Debug flags
CXXFLAGS+= -g -march=native

