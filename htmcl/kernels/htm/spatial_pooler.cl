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


kernel void
spatial_pooler_phase_2(global struct layer* l) {

    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

    unsigned int count_greater = 0;

    for (unsigned int i = 0; i < l->size; i++) {
        if (l->columns[i].overlap > c->overlap) {
            count_greater += 1;
        }
    }

    if (count_greater < l->desired_local_activity && c->overlap > 0.0f) {
        c->active = true;
    }
}


kernel void
spatial_pooler_phase_3(global struct htm* net, global struct layer* l) {

    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

    global list* cur_seg = c->segments;

    // Changing permanence of proximal synapses
    while (cur_seg != NULL) {
        global proximal_dendrite* d = (global proximal_dendrite*) cur_seg->data;

        global layer* input_layer = net->layers[d->connection->input_layer_id];
        global sdr* input = input_layer->sdr;

        global list* cur_syn = d->proximal_synapses;

        while (cur_syn != NULL) {
            global proximal_synapse* current_synapse = (global proximal_synapse*) cur_syn->data;

            bool value = get_value(input, current_synapse->input_position);

            if (value) {
                current_synapse->permanence = min(1.0f, current_synapse->permanence + l->permanence_inc);
            }
            else {
                current_synapse->permanence = max(0.0f, current_synapse->permanence - l->permanence_dec);
            }

            cur_syn = cur_syn->next;
        }

        cur_seg = cur_seg->next;
    }

    float max_duty_cycle = 0.0f;

    for (unsigned int i = 0; i < l->size; i++) {
        max_duty_cycle = max(l->columns[i].active_duty_cycle, max_duty_cycle);
    }

    float local_area_density = (float) l->desired_local_activity / (float) l->size;

    c->min_duty_cycle = max_duty_cycle * 0.01f;
    c->active_duty_cycle = (c->active_duty_cycle * (l->activity_duty_cycle - 1.0f) + (c->active ? 1.0f : 0.0f));
    c->overlap_duty_cycle = (c->overlap_duty_cycle * (l->overlap_duty_cycle - 1.0f) + (c->overlap > 0.0f ? 1.0f : 0.0f));
    c->boost = exp(-l->boost_strength * (c->active_duty_cycle - local_area_density));

    if (c->overlap_duty_cycle < c->min_duty_cycle) {
        cur_seg = c->segments;

        // Increasing permanence of proximal synapses
        while (cur_seg != NULL) {
            global proximal_dendrite* d = (global proximal_dendrite*) cur_seg->data;
            global list* cur_syn = d->proximal_synapses;

            while (cur_syn != NULL) {
                global proximal_synapse* current_synapse = (global proximal_synapse*) cur_syn->data;

                if (current_synapse->permanence > l->permanence_threshold) {
                    current_synapse->permanence = min(1.0f, current_synapse->permanence + 0.1f);
                }

                cur_syn = cur_syn->next;
            }

            cur_seg = cur_seg->next;
        }
    }
}
