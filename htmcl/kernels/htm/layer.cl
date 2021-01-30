typedef struct layer {

    unsigned int size_x;
    unsigned int size_y;

    unsigned int segment_activation_threshold;

    unsigned int segment_min_threshold;

    unsigned int cells_per_column;

    unsigned int distal_segments_per_cell;
    unsigned int apical_segments_per_cell;

    unsigned int synapses_per_distal_segment_limit;
    unsigned int synapses_per_apical_segment_limit;

    unsigned int initial_synapses_per_distal_segment;
    unsigned int initial_synapses_per_apical_segment;

    unsigned int max_new_synapses;

    float permanence_inc;
    float permanence_dec;

    float distal_permanence_inc;
    float distal_permanence_dec;

    float apical_permanence_inc;
    float apical_permanence_dec;

    float initial_proximal_permanence;
    float initial_distal_permanence;
    float initial_apical_permanence;

    float boost_inc;
    float boost_dec;
    float boost_strength;

    float overlap_threshold;

    float permanence_threshold;
    float permanence_limit;

    float proximal_decay;
    float distal_decay;
    float apical_decay;

    float activity_duty_cycle;
    float overlap_duty_cycle;

    bool learning;
    char __learning_padding[7];

    global struct cell* cells;
    global struct column* columns;

    global struct sdr* sdr;

} layer;


kernel void
get_layer_size_bytes(global unsigned int* result) {
    result[0] = sizeof(layer);
}


kernel void
prepare_layer_buffers(
        global layer* l,
        global struct cell* cells,
        global struct column* columns,
        global struct sdr* layer_sdr
) {

    l->cells = cells;
    l->columns = columns;
    l->sdr = layer_sdr;

}


kernel void
prepare_layer_primary_coefficients(
        global layer* l,

        unsigned int size_x,
        unsigned int size_y,
        unsigned int cells_per_column,
        unsigned int learning
) {

    l->size_x = size_x;
    l->size_y = size_y;
    l->cells_per_column = cells_per_column;
    l->learning = learning;

}


kernel void
prepare_layer_boost_coefficients(
        global layer* l,

        float boost_inc,
        float boost_dec,
        float boost_strength
) {

    l->boost_inc = boost_inc;
    l->boost_dec = boost_dec;
    l->boost_strength = boost_strength;

}


kernel void
prepare_layer_segment_coefficients(
        global layer* l,

        unsigned int segment_activation_threshold,
        unsigned int segment_min_threshold,
        unsigned int initial_synapses_per_apical_segment,
        unsigned int initial_synapses_per_distal_segment,
        unsigned int apical_segments_per_cell,
        unsigned int distal_segments_per_cell
) {

    l->segment_activation_threshold = segment_activation_threshold;
    l->segment_min_threshold = segment_min_threshold;
    l->initial_synapses_per_distal_segment = initial_synapses_per_distal_segment;
    l->initial_synapses_per_apical_segment = initial_synapses_per_apical_segment;
    l->apical_segments_per_cell = apical_segments_per_cell;
    l->distal_segments_per_cell = distal_segments_per_cell;

}


kernel void
prepare_input_layer(global layer* l) {

}

