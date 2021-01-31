typedef struct htm {

    unsigned int n_layers;
    unsigned int n_connections;

    global layer* global* layers;
    global layer_connection* connections;

} htm;


kernel void
get_htm_size_bytes(global unsigned int* result) {

    result[0] = sizeof(htm);

}


kernel void
init_htm(
    global htm* h,
    global void* layers,
    global layer_connection* connections
) {

    h->n_layers = 0;
    h->n_connections = 0;

    h->layers = (global layer* global*) layers;
    h->connections = connections;

}
