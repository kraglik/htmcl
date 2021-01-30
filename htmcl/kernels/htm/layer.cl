typedef struct layer {

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

    unsigned int duty_cycle_period;

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
    bool bursting;
    bool active;
    bool predicted;
    char __learning_padding[4];

    global struct cell* cells;
    global struct column* columns;
    global struct proximal_dendrite* proximal_dendrites;

    global struct sdr* sdr;

} layer;


kernel void
get_layer_size_bytes(global unsigned int* result) {
    result[0] = sizeof(layer);
}


kernel void
prepare_layer(global layer* l,
              unsigned int cells_per_column,
              unsigned int distal_segments_per_cell,
              unsigned int apical_segments_per_cell,
              unsigned int synapses_per_distal_segment_limit,
              unsigned int synapses_per_apical_segment_limit) {

}

kernel void
prepare_input_layer(global layer* l) {

}

