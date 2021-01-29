typedef struct synapse {

    unsigned int presynaptic_layer_id;
    unsigned int presynaptic_cell_id;

    unsigned int postsynaptic_layer_id;
    unsigned int postsynaptic_cell_id;

    unsigned long id;

    float permanence;

} synapse;

