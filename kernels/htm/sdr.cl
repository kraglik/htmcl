// Sparse Distributed Representation
typedef struct sdr {

    unsigned int n_dims;   // Number of dimensions that represents structure of given SDR
    unsigned int dims[3];  // Unused dimensions are equal to 1

    unsigned int state[];  // Bit array of size (dims[0] * dims[1] * dims[2])

} sdr;