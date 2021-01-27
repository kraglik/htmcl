struct neuron;
struct column;
struct synapse;
struct basal_dendrite;
struct proximal_dendrite;
struct apical_dendrite;
struct layer;
struct layer_connection;
struct htm_config;
struct htm;

typedef struct neuron {

    global struct column* column;
    unsigned long id;

    global list* distal_segments;
    global list* apical_segments;

    bool active, was_active, learning, was_learning, predictive, was_predictive;

} neuron;

typedef struct column {
    global struct neuron* neuron;
    global list* synapses;

    bool learning, was_learning, active, was_active, sequence;
} column;

typedef struct synapse {

} synapse;

typedef struct basal_dendrite {
    bool learning, was_learning, active, was_active, sequence;
} basal_dendrite;

typedef struct proximal_dendrite {

} proximal_dendrite;

typedef struct apical_dendrite {

} apical_dendrite;

typedef struct layer {

} layer;

typedef struct layer_connection {

} layer_connection;

typedef struct htm_config {

} htm_config;

typedef struct htm {

} htm;


/**********************************************************************************************************************/
/********************************************** NEURON RELATED FUNCTIONS **********************************************/


global basal_dendrite*
best_segment(global neuron* n) {

};


/**********************************************************************************************************************/
/****************************************** BASAL DENDRITE RELATED FUNCTIONS ******************************************/


void
basal_dendrite_step(global basal_dendrite* d) {
    d->was_active = d->active;
    d->active = false;
    d->was_learning = d->learning;
    d->learning = false;
    d->sequence = false;
}







kernel void
get_clheap_size(global unsigned int* out) {
    out[0] = sizeof(struct clheap);
}

typedef struct vec2 {
    unsigned int x, y;
    global volatile unsigned int* xs;
} vec2;


kernel void
test_allocations(global struct clheap* heap, global int* results) {

    unsigned int id = get_global_id(0);

    for (unsigned int i = 0; i < 128; i++) {

        global vec2* vec = (global vec2*) malloc(heap, sizeof(vec2));


        vec->x = id;
        vec->y = id + id;
        vec->xs = (global unsigned int*)malloc(heap, sizeof(unsigned int));

        results[id] = vec->x - vec->y;

        free(heap, vec->xs);
        free(heap, vec);

    }
}


kernel void
test_list_allocations(global struct clheap* heap, global int* results) {

    unsigned int id = get_global_id(0);

    for (unsigned int i = 0; i < 128; i++) {

        global list* l = new_node(heap, NULL, NULL);

        l->data = malloc(heap, sizeof(int));
        *((global int*)l->data) = id;

        for (unsigned int i = 0; i < 9; i++) {
            global int* data = (global int*) malloc(heap, sizeof(int));
            *data = id;
            append(heap, l, data);
        }

        results[id] = 0;

        global list* current = l;

        while (current != NULL && current->data != NULL) {
            results[id] += *((global int*)current->data);
            current = current->next;
        }

        free_local_list(heap, &l);

    }
}

kernel void
do_nothing(global struct clheap* heap) {

}


