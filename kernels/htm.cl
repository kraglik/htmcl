#ifndef HTM_CL
#define HTM_CL

struct neuron;
struct synapse;
struct dendrite;
struct layer;
struct htm_config;
struct htm;

struct layer_connection;

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
    global list * global * l,
    global void* item
) {

    global list* current = *l;
    global list* global* ptr_to_current = l;

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
get_clheap_size(global unsigned int* out) {
    out[0] = sizeof(struct clheap);
}

typedef struct vec2 {
    unsigned int x, y;
    global volatile unsigned int* xs;
} vec2;

kernel void
test_allocations(global struct clheap* heap, global int* results) {

    unsigned int id = get_global_id(0);

    for (unsigned int i = 0; i < 128; i++) {

        global vec2* vec = (global vec2*) malloc(heap, sizeof(vec2));


        vec->x = id;
        vec->y = id + id;
        vec->xs = (global unsigned int*)malloc(heap, sizeof(unsigned int));

        results[id] = vec->x - vec->y;

        free(heap, vec->xs);
        free(heap, vec);

    }
}

kernel void
test_list_allocations(global struct clheap* heap, global unsigned long* results) {

    unsigned int id = get_global_id(0);

    for (unsigned int i = 0; i < 1; i++) {

        global list* l = new_node(heap, NULL, NULL);

        l->data = malloc(heap, sizeof(unsigned int));
        *(global unsigned int*)l->data = id;

        global list* current = l;

        for (unsigned int i = 0; i < 9; i++) {
            current->next = new_node(heap, NULL, NULL);
            current->next->data = current->data;

            current = current->next;
        }

        current = l;

        results[id] = 0;

        while (current != NULL) {
            results[id] += *(global unsigned int*)current->data;
            current = current->next;
        }

        free_local_list(heap, &l);

    }
}

kernel void
do_nothing(global struct clheap* heap) {

}

#endif


