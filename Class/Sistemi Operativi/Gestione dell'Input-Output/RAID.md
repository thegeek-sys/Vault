---
Created: 2024-11-11
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Dischi RAID
RAID è l’acronimo di *Redundant Arrays of Indipendent Disks*. In alcuni casi, si hanno a disposizione più dischi fisici ed è possibile trattarli **separatamente** (es. Windows li mostrerebbe esplicitamente come dischi diversi, in Linux si potrebbe dire che alcune directory sono in un disco altre su un altro) oppure si possono considerare più dischi fisici come un **unico disco**

### Dischi multipli
In Linux, il trattare diversi dischi separatamente è chiamato Linux LVM (Logical Volume Manager).
Permette di avere alcuni files/directory sono memorizzati su un disco, altri si un altro e a farlo ci pensa direttamente il kernel (l’utente può non occuparsi di decidere dove salvare i file)