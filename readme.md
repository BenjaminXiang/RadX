# RadX

GPU sorting shaders dedication from vRt project.

## What to be want to do

- Optimized sorting for NVIDIA RTX GPU's (and, probably, Volta GPU's)
- Remove outdated or extra codes
- Add new experimental features without rendering backend
- In future, add support for other possible architectures (Radeon VII, Navi, Ampere)
- Add support for Intel UHD Graphics 630 (if we will have time)
- CUDA compute cabability 7.5 interporability

## To do in test code (C++)

- Make headless sorting test
- Make C++ headers for sort
- Add simple benchmark
- Backdrop vendor detection

## First Results 

- Tested with 1 million, 8 million and 64 million elements
- Outperforms `std::sort()` up to 30x in RTX 2070 GPU' in Debug build, and 1-2x in Release builds
- Tested include fences, and some memory copy
- Built in least Visual Studio 2019
