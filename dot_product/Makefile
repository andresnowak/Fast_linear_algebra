# Compiler & flags
CXX := clang++
CXX_FLAGS := -std=c++17 -Wall -Wextra
CXX_LDFLAGS := -framework Foundation -framework Metal

# Files
TEST_SRC := test.mm
METAL_SRC := dot_product_mul.metal
METAL_LIB := dot_product.metallib
OBJECTIVE_CPP_SRC := main.mm dot_product_mul.mm
TARGET := dot_product

.PHONY: all clean run

all: $(TARGET)

# Build the metal libraries
$(METAL_LIB): $(METAL_SRC)
	xcrun -sdk macosx metal -c $^ -o - | \
	xcrun -sdk macosx metallib -o $@ -

$(TARGET): $(OBJECTIVE_CPP_SRC) $(METAL_LIB)
# $@ the target of the build and $^ is the complete list of prerequisites (so here our objective c file and metal lib)
	$(CXX) $(CXX_FLAGS) $(CXX_LDFLAGS) $(OBJECTIVE_CPP_SRC) $(TEST_SRC) -o $@

run: $(TARGET)
	./$(TARGET)

clean: 
	rm -f $(METAL_LIB) $(TARGET)