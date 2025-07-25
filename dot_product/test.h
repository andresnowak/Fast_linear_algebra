#include <vector>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#else
typedef void NSString; // Dummy declaration to keep the compiler happy
#endif

std::pair<float, double> test_dot_product(const std::vector<float> &a, const std::vector<float> &b, float (*reduce_function)(size_t n, float *out), int gridSizeValues[3], int threadGroupSizeValues[3], NSString *libName);