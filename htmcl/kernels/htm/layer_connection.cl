typedef struct layer_connection {

    unsigned int input_layer_id;
    unsigned int target_layer_id;

    unsigned int max_synapses_per_segment;
    unsigned int radius;

    float min_permanence;
    float max_permanence;
    float threshold_permanence;
    float permanence_inc;
    float permanence_dec;

} layer_connection;


kernel void
get_layer_connection_size_bytes(global unsigned int* result) {

    result[0] = sizeof(layer_connection);

}

