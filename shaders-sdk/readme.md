# Radix sort dedicated shaders

- Here is dedicated shader source for GPU sorting

## TO DO

- Reduce processing key count to 2 for NVIDIA RTX sorting
- Add dedicated buffer for caching these keys
- Use 16-bit packing instead of 8-bit directly when not available to store
- Transition to reference buffer model (read buffer key by reference, and do final permute)
