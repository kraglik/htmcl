typedef struct column {

    unsigned int x, y;
    unsigned long id;

    global cell* cells;
    global list* segments;

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

