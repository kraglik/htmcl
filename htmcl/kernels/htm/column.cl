typedef struct column {

    unsigned int x, y;
    unsigned long id;

    global struct cell* cell;
    global list* synapses;

    float overlap_min;
    float overlap;
    float synaptic_activation;
    float boost;

    bool learning, was_learning;
    bool active, was_active;
    bool sequence;

} column;


kernel void
get_column_size_bytes(global unsigned int* result) {
    result[0] = sizeof(column);
}


kernel void
spatial_pooler_phase_1(global struct layer* l) {

    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

//    for (unsigned int i = 0; i < l->)

}

