typedef struct column {

    unsigned long id;

    global cell* cells;
    global list* segments;

    float overlap_min;
    float overlap;
    float active_duty_cycle;
    float min_duty_cycle;
    float overlap_duty_cycle;
    float boost;

    bool learning, was_learning;
    bool active, was_active;
    bool sequence;
    bool predicted;
    bool bursting;

} column;


kernel void
get_column_size_bytes(global unsigned long* result) {
    result[0] = sizeof(column);
}

void
column_step(
    global layer* l,
    global column* c
) {
    c->was_active = c->active;
    c->was_learning = c->learning;
    c->predicted = false;
    c->bursting = false;

    c->active = false;
    c->learning = false;

    for (unsigned int i = 0; i < l->cells_per_column; i++) {
        cell_step(l, &c->cells[i]);
    }
}


global struct cell*
get_cell_from_column(
    global struct column* col,
    unsigned int cell_pos
) {
    return col->cells + cell_pos;
}

