from memory import UnsafePointer, memset_zero
from random import rand, randint


struct Matrix[Type: DType](Copyable, Movable, Writable):
    var data: UnsafePointer[SIMD[Type, 1]]
    var rows: Int
    var cols: Int

    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        var data = UnsafePointer[SIMD[Type, 1]].alloc(rows * cols)
        rand(data, rows * cols, min=0, max=10)
        return Self(data, rows, cols)

    @staticmethod
    fn randint(rows: Int, cols: Int) -> Self:
        var temp = UnsafePointer[SIMD[DType.int8, 1]].alloc(rows * cols)
        randint(temp, rows * cols, 1, 10)
        var data = UnsafePointer[SIMD[Type, 1]].alloc(rows * cols)

        for i in range(rows):
            for j in range(cols):
                data[i + j * cols] = SIMD[Type, 1]((temp[i + j * cols]))

        return Self(data, rows, cols)

    fn __init__(out self, rows: Int, cols: Int):
        self.data = UnsafePointer[SIMD[Type, 1]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

        self.rows = rows
        self.cols = cols

    fn __init__(
        out self, data: UnsafePointer[SIMD[Type, 1]], rows: Int, cols: Int
    ):
        self.data = data
        self.rows = rows
        self.cols = cols

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        self.rows = existing.rows
        self.cols = existing.cols

    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.rows = existing.rows
        self.cols = existing.cols

    fn __getitem__(self, y: Int, x: Int) -> Scalar[Type]:
        return self.data.load[width=1](y * self.cols + x)

    fn __setitem__(self, y: Int, x: Int, value: Scalar[Type]):
        self.data.store[width=1](y * self.cols + x, value)

    fn load[width: Int](self, y: Int, x: Int) -> SIMD[Type, width]:
        return self.data.load[width=width](y * self.cols + x)

    fn store[width: Int](self, y: Int, x: Int, value: SIMD[Type, width]):
        self.data.store[width=width](y * self.cols + x, value)

    fn __del__(owned self):
        self.data.free()

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[")
        for i in range(self.rows):
            writer.write("[")
            for j in range(self.cols):
                writer.write(self[i, j])
                if j < self.cols - 1:
                    writer.write(", ")
            writer.write("]")
            if i < self.rows - 1:
                writer.write("\n")
        writer.write("]")

    fn __str__(self) -> String:
        var output = String()
        self.write_to(output)
        return output^
