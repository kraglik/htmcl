typedef struct htm {

    unsigned int n_layers;
    unsigned int n_connections;

    global layer* global* layers;
    global layer_connection* global* connections;

} htm;


kernel void
get_htm_size_bytes(global unsigned long* result) {

    result[0] = sizeof(htm);

}


kernel void
init_htm(
    global htm* h,
    global void* layers,
    global void* connections
) {

    h->n_layers = 0;
    h->n_connections = 0;

    h->layers = (global layer* global*) layers;
    h->connections = (global layer_connection* global*) connections;

}


kernel void
set_htm_connection(
    global htm* h,
    global layer_connection* connection,
    unsigned int conn_id
) {
    h->connections[conn_id] = connection;
}

kernel void
set_htm_layer(
    global htm* h,
    global layer* layer,
    unsigned int layer_id
) {
    h->layers[layer_id] = layer;
}
