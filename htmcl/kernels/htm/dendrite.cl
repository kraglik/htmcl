typedef struct dendrite {

    bool learning, was_learning;
    bool active, was_active;
    bool sequence;
    char __padding[3];
    global list* synapses;

} dendrite;


void
basal_dendrite_step(global dendrite* d) {
    d->was_active = d->active;
    d->active = false;
    d->was_learning = d->learning;
    d->learning = false;
    d->sequence = false;
}
