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

} column;


kernel void
get_column_size_bytes(global unsigned long* result) {
    result[0] = sizeof(column);
}

