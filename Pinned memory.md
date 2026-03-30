---
Class: "[[Multicore]]"
Related:
---
---
## DMA
DMA (Direct Memory Access) hardware is a unit specialized to transfer a number of bytes requested by OS between physical memory address space regions (some can be mapped to I/O memory allocations), and is used by `cudaMemcpy()` for better efficiency, as it frees CPU for other tasks and uses system interconnection (e.g. PCIe)

![[Pasted image 20260330125245.png|300]]

---
## Virtual memory management
Modern systems use virtual memory management to map many virtual memory spaces to a single physical memory and virtual addresses (pointer values) are translated into physical addresses

>[!warning] Not all variables and data structures are always in the physical memory
>In fact each virtual address space is divided into pages that are mapped into and out of the physical memory and virtual memory pages can be mapped out of the physical memory (*page-out*) to make room
>
>Whether or not a variable is in the physical memory is checked at address translation time

### Data transfers and virtual memory
