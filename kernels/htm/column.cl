typedef struct column {

    global struct cell* cell;
    global list* synapses;

    bool learning, was_learning;
    bool active, was_active;
    bool sequence;

} column;

