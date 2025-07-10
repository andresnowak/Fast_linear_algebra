# Constraint for mR, nR block is (mR/simdwidthof[Type]() * nR + mR/simdwidthof[Type]() + 1) <= 32, for m1 that has 32 ymm registers

def main():
    simd_width = 2
    nr_range = (32 - 1) * simd_width

    solutions = []

    for nr in range(nr_range):
        mr_max = nr_range // (nr + 1)
        for mr in range(mr_max):
            print(nr, mr)
            if mr % simd_width == 0 and (mr / simd_width * nr + mr/simd_width + 1) >= 24 and nr > 0 and mr > 0:
                solutions.append((mr, nr))

    print(solutions)

if __name__ == "__main__":
    main()