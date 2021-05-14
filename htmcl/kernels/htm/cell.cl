typedef struct cell {

    global struct column* column;
    unsigned long id;
    unsigned long next_synapse_id;

    global list* distal_segments;
    global list* apical_segments;

    bool active, was_active;
    bool learning, was_learning;
    bool predictive, was_predictive;

} cell;


global cell*
build_cell(
    global struct clheap* heap,
    global struct column* column,
    unsigned long id
) {
    global cell* c = (global cell*) malloc(heap, sizeof(cell));

    if (c != NULL) {

        c->id = id;
        c->column = column;
        c->distal_segments = NULL;
        c->apical_segments = NULL;

        c->active = false;
        c->was_active = false;
        c->predictive = false;
        c->was_predictive = false;
        c->learning = false;
        c->was_learning = false;

        c->next_synapse_id = 0;

    }

    return c;
}


kernel void
get_cell_size_bytes(global unsigned long* result) {
    result[0] = sizeof(cell);
}


global dendrite*
best_segment(global struct layer* l, global cell* c) {

    unsigned int best_activation = 0;
    global dendrite* best_segment = NULL;

    global list* cur_seg = c->distal_segments;

    // For segment in cell.distal_segments
    while (cur_seg != NULL) {
        global dendrite* current_segment = (global dendrite*) cur_seg->data;

        unsigned int segment_activation = 0;
        global list* cur_syn = current_segment->synapses;

        // For synapse in segment.synapses
        while (cur_syn != NULL) {
            global synapse* current_synapse = (global synapse*) cur_syn->data;

            // If synapse is active and presynaptic cell is also active
            if (current_synapse->permanence >= l->permanence_threshold
                && l->cells[current_synapse->presynaptic_cell_id].active) {

                segment_activation += 1;
            }

            cur_syn = cur_syn->next;
        }

        // Replace current best if this segment is better
        if (segment_activation > best_activation) {
            best_activation = segment_activation;
            best_segment = current_segment;
        }

        cur_seg = cur_seg->next;
    }

    return best_segment;
};


bool
cell_predicted(global cell* c) {
    return c->was_predictive && c->active;
}


void
dendrite_step(global layer* l, global dendrite* d);


void
cell_step(global layer* l, global cell* c) {

    c->was_predictive = c->predictive;
    c->was_learning = c->learning;
    c->was_active = c->active;

    c->predictive = false;
    c->learning = false;
    c->active = false;

    global list* cur_seg = c->distal_segments;

    while (cur_seg != NULL) {
        global dendrite* d = (global dendrite*) cur_seg->data;

        dendrite_step(l, d);

        cur_seg = cur_seg->next;
    }
}
