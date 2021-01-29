typedef enum input_layer_type {
    HTM_LAYER, SDR_LAYER
} input_layer_type;


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

    input_layer_type input_layer_type;

} layer_connection;
