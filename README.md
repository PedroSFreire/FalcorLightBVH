# Two Level Light BVH
 This repository contains the code and documentation for my master’s thesis on implementing a Two-Level Light Bounding Volume Hierarchy (BVH) to optimize ray tracing in complex 3D scenes with animated light meshes.
 Abstract:
 To render high-quality scenes with ray tracing we sample light sources at each ray intersection in
a process called next-event estimation. To do this many lights have to be considered, however, to
achieve real-time rendering not all lights can be sampled. In this thesis, we implement an algorithm
for accelerating importance sampling. This algorithm uses a two-level BVH to accelerate sampling.
The main advantage of the implemented algorithm is the ability to have dynamic lights without losing
excessive accuracy or performance. The algorithm rebuilds the top-level structure in the CPU allowing it
to retain its accuracy and refits only the bottom-level structures in need of updating on the GPU. The CPU
rebuild is a relatively costly operation, to avoid excessive performance loss it is done asynchronously only
being used in the next frame. We test several quality metrics as well as frame times, our implementation
didn’t yield the expected gain in quality for those quality metrics, however, close-up pictures showed an
improvement in some areas where moving lights start the animation and at their location. Performance
decreased an expected amount showing an increase in frame times of around 7%.

PDF of the thesis and the shortpaper published at wscg also in the rep.



 ![plot](./sample.png)

 
The implementation is done on top of falcor 5.2 https://github.com/NVIDIAGameWorks/Falcor
  
## Citation
If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

```bibtex
@inproceedings{lang2024web,
  title={Web-Based Flow Visualization in Quotient Space and S 3},
  author={Lang, Christian and Miftari, Egzon and Albers, Peter and Sadlo, Filip},
  booktitle={2024 IEEE 17th Pacific Visualization Conference (PacificVis)},
  pages={42--51},
  year={2024},
  organization={IEEE}
}
```
