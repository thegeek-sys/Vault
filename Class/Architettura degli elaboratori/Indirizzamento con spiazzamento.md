---
Created: 2024-03-24
Class: "[[Architettura degli elaboratori]]"
Related:
  - "[[Codifica delle istruzioni]]"
Completed:
---
---
## Introduction
Nonostante il mantenere tutte le istruzioni a 32 bit semplifica l’hardware, spesso ci torna utile (come per le istruzioni di jump) passare una costnate o un indirizzo a 32 bit all’interno di un’istruzione.

Infatti per caricare ad esempio un indirizzo di memoria a 32 bit all’interno di un registro vengono eseguite le seguenti operazioni:
1. Innanzitutto attraverso l’istruzione `lui` (load upper immediate) vengono caricati i primi 16 bit nei primi 16 bit (da sinistra) del registro
2. Infine viene fatto un or immediato `ori` tra il registro in cui ho eseguito il load upper e i 16 bit rimanenti dell’indirizzo in modo tale da poter scrivere nei 16 bit rimanenti

>[!note]
>`lui $t0,0xFF`
> 
> | 0000 0000 1111 1111 | 0000 0000 0000 0000 |
> |---|---|
>`ori $t0,$t0,2304  # 0000 1001 0000 0000`
> 
> | 0000 0000 1111 1111 | 0000 1001 0000 0000 |
> |---|---|

