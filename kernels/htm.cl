#ifndef HTM_CL
#define HTM_CL

#include "htm.h"
#include "kma.cl"

kernel void
get_clheap_size(global unsigned int* out) {
    out[0] = sizeof(struct clheap);
}

typedef struct vec2 {
    unsigned int x, y;
} vec2;

kernel void
test_allocations(global struct clheap* heap, global int* results) {

    global vec2* vec = (global vec2*) malloc(heap, sizeof(vec2));
    unsigned int id = get_global_id(0);

    vec->x = id;
    vec->y = id + id;

    results[id] = vec->x - vec->y;

    free(heap, vec);
}

#endif


