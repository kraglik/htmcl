kernel void
temporal_memory_phase_1(
    global struct layer* l
) {
    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

    if (!c->active) {
        return;
    }

    for (unsigned int i = 0; i < l->cells_per_column; i++) {
        global cell* n = &c->cells[i]; // n for neuron

        if (n->was_predictive) {
            n->predicted = true;
            n->active = true;
            c->predicted = true;
        }
    }

    if (!c->predicted) {
        c->bursting = true;

        for (unsigned int i = 0; i < l->cells_per_column; i++) {
            global cell* n = &c->cells[i];

            n->bursting = true;
            n->active = true;
        }
    }
}


kernel void
temporal_memory_phase_2(
    global struct layer* l
) {
    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

    if (!c->active) {
        return;
    }


}
