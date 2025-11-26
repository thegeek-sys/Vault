---
Class: "[[Multicore]]"
Related:
---
---
## CPU vs GPU

>[!info] Latecy vs. throughput
>- latency → how much do I take to execute a task
>- throughput → how many tasks I can execute for time unit

>[!hint] Understanding the images
>The more the ALU is complex, the more space it occupies inside the CPU/GPU

### CPU
![[Pasted image 20251125171918.png|350]]

CPUs have a **latency oriented design** which results in: high clock frequency, large caches (convert long latency memory access to short latency cache accesses), sophisticated control (branch prediction for reduced branch latency, out-of-order execution, etc.), and a powerful ALU (where the computation is done, to reduce operation latency).

So the CPUs are designed to reduce as much as possible the instruction’s execution time

### GPU
![[Pasted image 20251125172404.png|350]]

GPUs have a **throughput oriented design** which results in: moderate clock frequency, small caches, simple control (no branch prediction, in-order execution), and energy efficient ALUs (many, long latency but heavily pipelined for high throughput)

But to tollerate latencies, they require massive number of threads and use high bandwidth interfaces for memory and host data exchange (memory accesses are much faster in GPU rather than CPU, as CPU uses DRAM while GPU has a dedicated one)

>[!hint] Design choices
>The control unit and caches are very small, to be able to have much more execution units
>
>As they have many cores it’s possible to hide memory accesses, in fact, while the data is transferred, it’s possible to execute other instructions

---
## Architecture of a CUDA-capable GPU
![[Pasted image 20251125172610.png]]

---
## Application benefits from both CPU and GPU
CPUs for sequential parts where latency mattes (CPUs can be 10+X faster than GPUs for sequential code) and GPUs for parallel parts where throughput matters (GPUs cab be 10+X faster than CPUs for parallel code)

### CPU-GPU architecture
![[Pasted image 20251125172957.png]]

(a) and (b) represent discrete GPU solutions, with a CPU-integrated memory controller in (b). Diagram (c) corresponds to integrated CPU-GPU solutions, such as the AMD’s Accelerated Processing Unit (APU) chips.

>[!warning] GPU programming caveats
>GPU program deployment has a characteristic that can be considered a major obstacle: GPU and host memories are typically disjoint, requiring explicit (or implicit, depending on the development platform) data transfer between the two
>
>A second characteristic of GPU computation is that GPU devices may not adhere to the same floating-point representation and accuracy standards as typical CPUs.

---
## GPU software development platforms
### CUDA
**CUDA** (*Compute Unified Device Architecture*) provides two sets of APIs (a low and a higher level one) and it is available freely for Windows, MacOS X and Linux operating systems.

Major drawback: NVidia hardware only (even though now there are tools to run CUDA code on AMD GPUs)

### HIP
HIP is the AMD’s equivalent of CUDA and there are tools provided to convert CUDA to HIP code (HIPIFY)

### OpenCL
**OpenCL** (*Open Computing Language*) is an open standard for writing programs that can execute across a variety of heterogeneous platforms that include GPUs. CPUs, DSPs or other processor.

OpenCL is supported by both NVidia and AMD and is the primary development platform for AMD GPUs. OpenCL’s programming model matched closely the one offered by CUDA

### OpenACC
**OpenACC** is an open specification for an API that allows the use of compiler directives (e.g. `#pragma acc`, in a similar fashion to OpenMP) to automatically map computations to GPUs or multicore chips, according to a programmer’s hints