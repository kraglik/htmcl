typedef struct layer {

    unsigned int active_columns_num;
    unsigned int segment_activation_threshold;
    unsigned int segment_min_threshold;
    unsigned int cells_per_column;
    unsigned int basal_segments_per_cell;
    unsigned int apical_segments_per_cell;
    unsigned int synapses_per_basal_segment_limit;
    unsigned int synapses_per_apical_segment_limit;
    unsigned int initial_synapses_per_basal_segment;
    unsigned int initial_synapses_per_apical_segment;
    unsigned int max_new_synapses;
    unsigned int duty_cycle_period;

    float permanence_inc;
    float permanence_dec;

    float basal_permanence_inc;
    float basal_permanence_dec;

    float apical_permanence_inc;
    float apical_permanence_dec;

    float initial_proximal_permanence;
    float initial_basal_permanence;
    float initial_apical_permanence;

    float boost_inc;
    float overlap_threshold;
    float permanence_threshold;
    float permanence_limit;
    float proximal_decay;
    float basal_decay;
    float apical_decay;
    float boost_strength;

    bool learning;
    char __learning_padding[7];

    global cell* cells;
    global column* columns;

    global sdr* sdr;

} layer;

