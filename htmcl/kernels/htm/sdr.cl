// Sparse Distributed Representation
typedef struct sdr {

    unsigned long size;

    bool state[];  // Bool array of size (dims[0] * dims[1] * dims[2])
                   // Since it will be accessed simultaneously by multiple threads,
                   // it is way more simple to keep it as just an array of booleans.

} sdr;

kernel void
get_sdr_size_bytes(global unsigned int* result, unsigned int size) {

    result[0] = sizeof(sdr) + sizeof(bool) * size;

}

kernel void
init_sdr(global sdr* sdr,
         unsigned int size) {
    sdr->size = size;

    for (unsigned int i = 0; i < size; i++) {
        sdr->state[i] = false;
    }
}


bool
get_value(global sdr* s, unsigned int position) {
    return s->state[position];
}

void
set_value(global sdr* s, unsigned int position, bool value) {
    s->state[position] = value;
}

