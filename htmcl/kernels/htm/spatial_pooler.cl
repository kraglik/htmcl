kernel void
spatial_pooler_phase_1(global struct layer* l) {

    unsigned long column_id = get_global_id(0);
    global column* c = &l->columns[column_id];

//    for (unsigned int i = 0; i < l->)

}