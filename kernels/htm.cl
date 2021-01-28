struct cell;
struct column;
struct synapse;
struct basal_dendrite;
struct proximal_dendrite;
struct apical_dendrite;
struct layer;
struct layer_connection;
struct htm_config;
struct htm;

typedef struct cell {

    global struct column* column;
    unsigned long id;

    global list* distal_segments;
    global list* apical_segments;

    bool active, was_active, learning, was_learning, predictive, was_predictive;

} cell;

typedef struct column {
    global struct cell* cell;
    global list* synapses;

    bool learning, was_learning, active, was_active, sequence;
} column;

typedef struct synapse {

} synapse;

typedef struct basal_dendrite {
    bool learning, was_learning, active, was_active, sequence;
} basal_dendrite;

typedef struct proximal_dendrite {

} proximal_dendrite;

typedef struct apical_dendrite {

} apical_dendrite;

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

} layer;

typedef struct layer_connection {

} layer_connection;

typedef struct htm_config {

} htm_config;

typedef struct htm {

} htm;


/**********************************************************************************************************************/
/*********************************************** CELL RELATED FUNCTIONS ***********************************************/


global basal_dendrite*
best_segment(global cell* n) {

};


/**********************************************************************************************************************/
/****************************************** BASAL DENDRITE RELATED FUNCTIONS ******************************************/


void
basal_dendrite_step(global basal_dendrite* d) {
    d->was_active = d->active;
    d->active = false;
    d->was_learning = d->learning;
    d->learning = false;
    d->sequence = false;
}



kernel void
do_nothing(global struct clheap* heap) {

}


