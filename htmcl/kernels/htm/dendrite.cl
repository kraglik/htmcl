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
dendrite_step(global layer* l, global dendrite* d) {
    d->was_active = d->active;
    d->active = false;
    d->was_learning = d->learning;
    d->learning = false;
    d->sequence = false;

    global list* cur_syn = d->synapses;

    unsigned int activation = 0;

    while (cur_syn != NULL) {
        global synapse* current_synapse = (global synapse*) cur_syn->data;

        if (current_synapse->permanence >= l->permanence_threshold
            && cell_was_active(get_cell_from_layer(l, current_synapse->presynaptic_cell_id))) {

            activation += 1;
        }

        cur_syn = cur_syn->next;
    }

    if (activation >= l->segment_activation_threshold) {
        d->active = true;
    }
}
