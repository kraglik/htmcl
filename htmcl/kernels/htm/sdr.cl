// Sparse Distributed Representation
typedef struct sdr {

    unsigned int n_dims;   // Number of dimensions that represents structure of given SDR
    unsigned int dims[3];  // Unused dimensions are equal to 1

    bool state[];  // Bool array of size (dims[0] * dims[1] * dims[2])
                   // Since it will be accessed simultaneously by multiple threads,
                   // it is way more simple to keep it as just an array of booleans.

} sdr;


kernel void
get_sdr_size_bytes(global unsigned int* result,
                   unsigned int size_x,
                   unsigned int size_y,
                   unsigned int size_z) {

    result[0] = sizeof(sdr) + sizeof(bool) * size_x * size_y * size_z;

}


kernel void
init_sdr(global sdr* sdr,
         unsigned int size_x,
         unsigned int size_y,
         unsigned int size_z,
         unsigned int n_dims) {

    sdr->n_dims = n_dims;
    sdr->dims[0] = size_x;
    sdr->dims[1] = size_y;
    sdr->dims[2] = size_z;

    for (unsigned int i = 0; i < (size_x * size_y * size_z); i++) {
        sdr->state[i] = false;
    }

}


bool
get_value_1d(global sdr* s, unsigned int position) {
    return s->state[position];
}

bool
get_value_2d(global sdr* s, unsigned int p1, unsigned int p2) {
    return s->state[p2 * s->dims[1] + p1];
}

bool
get_value_3d(global sdr* s, unsigned int p1, unsigned int p2, unsigned int p3) {
    return s->state[(s->dims[2] * p3 + p2) * s->dims[1] + p1];
}

void
set_value_1d(global sdr* s, unsigned int position, bool value) {
    s->state[position] = value;
}

void
set_value_2d(global sdr* s, unsigned int p1, unsigned int p2, bool value) {
    s->state[p2 * s->dims[1] + p1] = value;
}

void
set_value_3d(global sdr* s, unsigned int p1, unsigned int p2, unsigned int p3, bool value) {
    s->state[(s->dims[2] * p3 + p2) * s->dims[1] + p1] = value;
}
