typedef struct proximal_synapse {

    unsigned int input_position;
    float permanence;

} proximal_synapse;


global proximal_synapse*
build_proximal_synapse(global struct clheap* heap,
                       float permanence,
                       unsigned int position) {

    global proximal_synapse* s = (global proximal_synapse*) malloc(heap, sizeof(proximal_synapse));

    s->input_position = position;
    s->permanence = permanence;

    return s;

}
