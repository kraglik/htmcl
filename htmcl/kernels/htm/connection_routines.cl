kernel void
randomly_connect_layers(
    global struct clheap* heap,
    global htm* net,
    global layer* input,
    global layer* target,
    global random_seed* random,
    unsigned int connection_id,
    float connection_probability
) {

    unsigned int id = get_global_id(0);

    global layer_connection* conn = net->connections[connection_id];

    global column* c = &target->columns[id];
    global proximal_dendrite* segment = build_proximal_dendrite(heap, conn);

    prepend(heap, &c->segments, segment);

    for (unsigned int i = 0; i < input->size; i++) {
        if (next_f32(random) <= conn->probability) {

            float permanence = next_f32(random);
            global proximal_synapse* s = build_proximal_synapse(heap, permanence, i);

            prepend(heap, &segment->proximal_synapses, s);
        }
    }

}