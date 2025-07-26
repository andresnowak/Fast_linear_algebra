# Compiler & flags
CXX := clang++
CXX_FLAGS := -arch arm64 -std=c++17 -Wall -Wextra -x objective-c++ -DMACOSX_DEPLOYMENT_TARGET=10.15 -fopenmp
CXX_LDFLAGS := -framework Foundation -framework Metal

# Files
TEST_SRC := test.mm
METAL_SRC := dot_product_mul.metal dot_product_mul_reduce.metal dot_product_mul_reduce_atomic.metal dot_product_mul_tree_reduce.metal
METAL_LIBS := dot_product_mul.metallib dot_product_mul_reduce.metallib dot_product_mul_reduce_atomic.metallib dot_product_mul_tree_reduce.metallib
OBJECTIVE_CPP_SRC := main.mm dot_product.mm
TARGET := dot_product

.PHONY: all clean run

all: $(TARGET) $(METAL_LIBS)

# Build the metal libraries

%.metallib: %.metal
	xcrun -sdk macosx metal -c $< -o - | \
	xcrun -sdk macosx metallib -o $@ -

$(TARGET): $(OBJECTIVE_CPP_SRC) $(TEST_SRC) $(METAL_LIBS) 
# $@ the target of the build and $^ is the complete list of prerequisites (so here our objective c file and metal lib)
	$(CXX) $(CXX_FLAGS) $(CXX_LDFLAGS) $(OBJECTIVE_CPP_SRC) $(TEST_SRC) -o $@

run: $(TARGET)
	./$(TARGET)

clean: 
	rm -f $(METAL_LIBS) $(TARGET)