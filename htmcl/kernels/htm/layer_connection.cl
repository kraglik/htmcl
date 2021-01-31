typedef struct layer_connection {

    unsigned int input_layer_id;
    unsigned int target_layer_id;
    global layer* layer;

} layer_connection;


kernel void
get_layer_connection_size_bytes(global unsigned int* result) {

    result[0] = sizeof(layer_connection);

}

