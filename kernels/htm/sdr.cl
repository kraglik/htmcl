// Sparse Distributed Representation
typedef struct sdr {

    unsigned int n_dims;   // Number of dimensions that represents structure of given SDR
    unsigned int dims[3];  // Unused dimensions are equal to 1

    bool state[];  // Bool array of size (dims[0] * dims[1] * dims[2])
                   // Since it will be accessed simultaneously by multiple threads,
                   // it is way more simple to keep it as just an array of booleans.

} sdr;


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
