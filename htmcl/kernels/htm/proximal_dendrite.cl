typedef struct proximal_dendrite {

    global struct layer_connection* connection;
    global struct list* proximal_synapses;

} proximal_dendrite;


global proximal_dendrite*
build_proximal_dendrite(global struct clheap* heap, global struct layer_connection* conn) {

    global proximal_dendrite* d = (global proximal_dendrite*) malloc(heap, sizeof(proximal_dendrite));

    d->connection = conn;
    d->proximal_synapses = NULL;

    return d;

}

