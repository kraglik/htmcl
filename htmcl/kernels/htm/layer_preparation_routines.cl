kernel void
prepare_layer_columns(global layer* l) {

    unsigned int id = get_global_id(0);

    global cell* cells = &l->cells[id * l->cells_per_column];
    global column* c = &l->columns[id];

    c->overlap_min = l->overlap_threshold;
    c->active_duty_cycle = 1.0f;
    c->overlap_duty_cycle = 1.0f;
    c->min_duty_cycle = 1.0f;
    c->boost = 1.0f;

    c->id = id;
    c->cells = cells;
    c->segments = NULL;

    c->learning = false;
    c->was_learning = false;
    c->active = false;
    c->was_active = false;
    c->sequence = false;

    for (unsigned int i = 0; i < l->cells_per_column; i++) {

        global cell* current_cell = &cells[i];

        current_cell->id = id * l->cells_per_column + i;
        current_cell->distal_segments = NULL;
        current_cell->apical_segments = NULL;
        current_cell->column = c;

        current_cell->active = false;
        current_cell->was_active = false;
        current_cell->predictive = false;
        current_cell->was_predictive = false;
        current_cell->learning = false;
        current_cell->was_learning = false;

    }

}

kernel void
randomly_connect_layer(global layer* l) {

    

}

