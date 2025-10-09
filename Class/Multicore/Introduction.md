---
Created: 2025-10-08
Related:
Class: "[[Multicore]]"
---
---
## Index
- [[#How will we write parallel programs?|How will we write parallel programs?]]
- [[#Type of parallel systems|Type of parallel systems]]
- [[#Concurrent vs. parallel vs. distributed|Concurrent vs. parallel vs. distributed]]
- [[#Why do we care about the hardware?|Why do we care about the hardware?]]
	- [[#Why do we care about the hardware?#Von Neumann architecture|Von Neumann architecture]]
		- [[#Von Neumann architecture#Its bottleneck|Its bottleneck]]
---
## How will we write parallel programs?
We will write programs that are explicitly parallel, using four different extentions of the C APIs:
- **Message-Passing Interface** (*MPI*) `[Library]`
- **Posix Threads** (*Pthreads*) `[Library]`
- **OpenMP** `[Library + Compiler]`
- **CUDA** `[Library + Compiler]`

Higher-level libraries exist, but they trade-off ease-of-use for performance (ig. the more performing/efficient you want to be, the more you need to suffer)

---
## Type of parallel systems
We can distinguish two different kind of system based on how they manage memory:
- **shared memory** → the cores can share access to the computer’s memory, and they have to be coordinated to examine and update shared memory locations
- **distributed memory** → each core has its own private memory, and the cores must communicate explicitly by sending messages across a network

![[Pasted image 20251009165142.png]]

But we can also distinguish two different kind of system based on which instruction each core can execute:
- **Multiple-Instruction Multiple-Data** (*MIMD*) → each core has its own control units and can work independently from the others
- **Single-Instruction Multiple-Data** (*SIMD*) → cores share the control units (they must all execute the same instruction, or stay idle)

>[!info] NB
>In SIMD it’s not necessary that all the cores execute the same instruction, you can assign different task to different groups of cores. Despite that it still has a low efficiency

![[Pasted image 20251008173131.png|350]]

---
## Concurrent vs. parallel vs. distributed
There isn’t a complete agreement on the definition but:
- **concurrent** → multiple tasks can be in progress at any time (concurrent programs can be serial, eg. multitasking os running on a single core)
- **parallel** → multiple tasks cooperate closely to solve a problem (tightly coupled, cores share the memory or are connected through a fast network)
- **distributed** → a program might need to cooperate with other programs to solve a problem (more loosely coupled than parallel)

But parallel and distributed are concurrent

---
## Why do we care about the hardware?
It is possible to abstract it when we are programming, but if you want to write efficient code, is better to know on which hardware you are running on, and to optimize for that

### Von Neumann architecture
![[Pasted image 20251008173800.png|center|300]]

- **main memory** → collection of locations, where each one has an address (used to access that location) and some content (data or instruction)
- **CPU** → control unit (decides which instructions execute) and datapath (executes the instructions). The state of an executing program is stored in registers (eg. PC)
- **interconnect** → used to transfer data between CPU and memory. Traditionally a bus, but can be much more complex

#### Its bottleneck
A Von Neumann machine executes one instruction at a time, each operating on a few pieces of data (stored in registers). CPU can read (fetch) data from memory, or write (store) data to it

Separation of CPU and memory is known as *Von Neumann bottleneck*, and the interconnect determines the rate at which data is trasnferred

>[!question] How much does that cost?
>![[Pasted image 20251008174329.png]]
>
>![[Pasted image 20251008174344.png]]

