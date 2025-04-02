class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other, 0)

        real = self.real + other.real
        dual = self.dual + other.dual

        return Dual(real, dual)

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Dual):
            return Dual(self.real * other, self.dual * other)

        real = self.real * other.real
        dual = self.real * other.dual + self.dual * other.real # dual * dual = 0

        return Dual(real, dual)

    __rmul__ = __mul__


print((3 * Dual(3, 1) + Dual(3, 1)).dual)
