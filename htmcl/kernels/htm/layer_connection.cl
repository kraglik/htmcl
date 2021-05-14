typedef struct layer_connection {

    unsigned int input_layer_id;
    unsigned int target_layer_id;

    float probability;
    unsigned int __padding;

} layer_connection;


kernel void
get_layer_connection_size_bytes(global unsigned int* result) {

    result[0] = sizeof(layer_connection);

}

kernel void
init_connection(global layer_connection* conn,
                unsigned int input_layer_id,
                unsigned int target_layer_id,
                float probability) {

    conn->input_layer_id = input_layer_id;
    conn->target_layer_id = target_layer_id;
    conn->probability = probability;

}
