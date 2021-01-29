typedef struct htm {

    unsigned int n_layers;
    unsigned int n_connections;

    global layer* layers;
    global layer_connection* connections;

} htm;


kernel void
init_htm(global htm* h) {

}
