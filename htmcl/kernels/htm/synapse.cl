typedef struct synapse {

    unsigned int presynaptic_cell_id;
    unsigned int postsynaptic_cell_id;

    unsigned int cell_synapse_id;

    float permanence;

} synapse;


global synapse*
build_synapse(
    global struct clheap* heap,
    unsigned int presynaptic_cell_id,
    unsigned int postsynaptic_cell_id,
    unsigned int cell_synapse_id,
    float permanence
) {

    global synapse* s = (global synapse*) malloc(heap, sizeof(synapse));

    if (s != NULL) {
        s->presynaptic_cell_id = presynaptic_cell_id;
        s->postsynaptic_cell_id = postsynaptic_cell_id;
        s->cell_synapse_id = cell_synapse_id;
        s->permanence = permanence;
    }

    return s;

}
