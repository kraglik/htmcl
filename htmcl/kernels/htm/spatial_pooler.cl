kernel void
spatial_pooler_phase_1(global struct htm* net, global struct layer* l) {

    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

    float overlap = 0.0f;

    global list* cur_seg = c->segments;

    while (cur_seg != NULL) {
        global proximal_dendrite* d = (global proximal_dendrite*) cur_seg->data;

        global layer* input_layer = net->layers[d->connection->input_layer_id];
        global sdr* input = input_layer->sdr;

        global list* cur_syn = d->proximal_synapses;

        while (cur_syn != NULL) {
            global proximal_synapse* current_synapse = (global proximal_synapse*) cur_syn->data;

            if (current_synapse->permanence < l->permanence_threshold)
                continue;

            bool value = get_value(input, current_synapse->input_position);

            if (value)
                overlap += 1.0f;

            cur_syn = cur_syn->next;
        }

        cur_seg = cur_seg->next;
    }

    if (overlap < c->overlap_min) {
        overlap = 0.0f;
    }

    overlap *= c->boost;
    c->overlap = overlap;
}