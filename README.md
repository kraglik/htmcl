## HTMCL

This repo contains an implementation of Hierarchical Temporal Memory in OpenCL.

TODO List:
- [x] Adapt [KMA](https://github.com/RSpliet/KMA) for use with PyOpenCL and test it
    - [x] Test it (something near 6 millions malloc calls per second)
    - [x] Fix that bug with code failing on Nvidia GPU. It was caused by wrong objects alignment in memory in case of 64-bit GPU.
- [x] Implement random
    - [x] Test it
- [x] Implement generic list
    - [x] Test it
- [x] Implement input layer
- [ ] Implement spatial pooler
- [ ] Implement temporal pooler
- [ ] Implement classifier
- [ ] Implement multi-layer architecture
- [ ] Write a proper documentation

