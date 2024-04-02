#include <stdio.h>
#include <cuda.h>
 

extern "C" {
    int maxmul() {
      size_t free_t, total_t;
      cudaMemGetInfo(&free_t, &total_t);
      return int(free_t/1024/1024);
    };

}
