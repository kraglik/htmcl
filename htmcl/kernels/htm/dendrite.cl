typedef struct dendrite {

    bool learning, was_learning;
    bool active, was_active;
    bool sequence;
    char __padding[3];
    global list* synapses;

} dendrite;


global dendrite*
build_dendrite(global struct clheap* heap) {

    global dendrite* d = (global dendrite*) malloc(heap, sizeof(dendrite));

    if (d != NULL) {

        d->learning = false;
        d->was_learning = false;

        d->active = false;
        d->was_active = false;
        d->sequence = false;
        d->synapses = NULL;

    }

    return d;
}


void
distal_dendrite_step(global layer* l, global dendrite* d) {
    d->was_active = d->active;
    d->active = false;
    d->was_learning = d->learning;
    d->learning = false;
    d->sequence = false;
}
