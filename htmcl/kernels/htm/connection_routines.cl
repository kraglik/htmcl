kernel void
randomly_connect_layers(
    global struct clheap* heap,
    global htm* h,
    global random_seed* random,
    global layer* input,
    global layer* target,
    unsigned int connection_id,
    float connection_probability
) {

    unsigned int id = get_global_id(0);

    global column* c = &target->columns[id];
    global layer_connection* conn = &h->connections[connection_id];

    global proximal_dendrite* segment = build_proximal_dendrite(heap, conn);

    append(heap, &c->segments, segment);

}