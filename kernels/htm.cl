#ifndef HTM_CL
#define HTM_CL

struct neuron;
struct synapse;
struct dendrite;
struct layer;
struct htm_config;
struct htm;

struct synapse_list;
struct neuron_list;
struct dendrite_list;
struct layer_list;

struct layer_connection;


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

kernel void
do_nothing(global struct clheap* heap) {

}

#endif


