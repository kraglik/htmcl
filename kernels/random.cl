typedef unsigned long random_seed;


unsigned int
next_uint32(global random_seed* random) {

    unsigned int id = get_global_id(0);

    unsigned long seed = random[id];

    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    random[id] = seed;

    unsigned int result = seed >> 16;

    return result;
}

unsigned long
next_uint64(global random_seed* random) {
    unsigned int id = get_global_id(0);

    unsigned long seed = random[id];

    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    random[id] = seed;

    return seed;
}

int
next_int32(global random_seed* random) {
    unsigned int id = get_global_id(0);

    unsigned long seed = random[id];

    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    random[id] = seed;

    int result = (unsigned int)(seed >> 16) - 2147483646;

    return result;
}

long
next_int64(global random_seed* random) {
    unsigned int id = get_global_id(0);

    unsigned long seed = random[id];

    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    random[id] = seed;

    long result = (long)(seed) - 9223372036854775806L;

    return result;
}

float
next_f32(global random_seed* random) {
    unsigned int id = get_global_id(0);
    unsigned int x = next_int32(random);

    return (float)x / 4294967295.0;
}


kernel void
test_random(global random_seed* random, global float* result) {
    for (unsigned int i = 0; i < 128; i++) {
        result[get_global_id(0)] = next_f32(random);
    }
}
