typedef struct list {
    global struct list* next;
    global void* data;
} list;


global list*
new_node(global struct clheap* heap, global list* next, global void* data) {
    global list* node = (global list*)malloc(heap, sizeof(list));

    node->next = next;
    node->data = data;

    return node;
}


void
remove_first_entry(
    global struct clheap* heap,
    global list* global * l,
    global void* item
) {

    global list* current = *l;
    global list* global * prev_next_ptr = l;

    while (current != NULL && current->next != NULL && current->next->data != item) {
        prev_next_ptr = &current->next;
        current = current->next;
    }

    if (current != NULL && current->next != NULL) {
        *prev_next_ptr = current->next->next;
        free(heap, current);
    }

}


void
remove(
    global struct clheap* heap,
    global list* global * l,
    global void* item
) {

    global list* current = *l;
    global list* global * prev_next_ptr = l;

    while (current != NULL && current->next != NULL) {

        if (current->next->data == item) {
            global list* to_remove = current->next;
            current->next = to_remove->next;
            free(heap, to_remove);
        }

        prev_next_ptr = &current->next;
        current = current->next;
    }

}


void
append(
    global struct clheap* heap,
    global list * l,
    global void* item
) {

    global list* current = l->next;
    global list* global* ptr_to_current = &l->next;

    while (current != NULL) {
        ptr_to_current = &current->next;
        current = current->next;
    }

    *ptr_to_current = new_node(heap, NULL, item);
}

void
insert_at(
    global struct clheap* heap,
    global list* global* l,
    global void* item,
    unsigned int position
) {

    global list* current = *l;
    global list* global* ptr_to_current = l;

    unsigned int i;

    for (i = 0; i < position && current != NULL; i++) {
        ptr_to_current = &current->next;
        current = current->next;
    }

    if (i == position) {
        *ptr_to_current = new_node(heap, *ptr_to_current, item);
    }

}


unsigned int
len(global list* l) {
    unsigned int length = 0;

    while (l != NULL) {
        length++;
        l = l->next;
    }

    return length;
}


global list*
pop_left(global list* global* l) {

    global list* node = *l;

    l = &node->next;

    return node;

}


global list*
pop_right(global list* global* l) {

    global list* current = *l;
    global list* global* ptr_to_current = l;

    while (current != NULL && current->next != NULL) {

        ptr_to_current = &current->next;
        current = current->next;

    }

    *ptr_to_current = NULL;

    return current;

}


void
free_list(global struct clheap *heap, global list* global *l) {
    global list* next = *l;

    while (next != NULL) {
        free(heap, next->data);
        global list* temp = next->next;
        free(heap, next);
        next = temp;
    }

    *l = NULL;
}


void
free_local_list(global struct clheap *heap, global list** l) {
    global list* next = *l;

    while (next != NULL) {
        free(heap, next->data);
        global list* temp = next->next;
        free(heap, next);
        next = temp;
    }

    *l = NULL;
}


kernel void
test_list_allocations(global struct clheap* heap, global int* results) {

    unsigned int id = get_global_id(0);

    for (unsigned int i = 0; i < 128; i++) {

        global list* l = new_node(heap, NULL, NULL);

        l->data = malloc(heap, sizeof(int));
        *((global int*)l->data) = id;

        for (unsigned int i = 0; i < 9; i++) {
            global int* data = (global int*) malloc(heap, sizeof(int));
            *data = id;
            append(heap, l, data);
        }

        results[id] = 0;

        global list* current = l;

        while (current != NULL && current->data != NULL) {
            results[id] += *((global int*)current->data);
            current = current->next;
        }

        free_local_list(heap, &l);

    }
}