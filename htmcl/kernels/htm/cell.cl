typedef struct cell {

    global struct column* column;
    unsigned long id;

    global list* distal_segments;
    global list* apical_segments;

    bool active, was_active;
    bool learning, was_learning;
    bool predictive, was_predictive;

} cell;


kernel void
get_cell_size_bytes(global unsigned long* result) {
    result[0] = sizeof(cell);
}


global dendrite*
best_segment(global cell* n) {

};

