---
Created: 2024-04-15
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
$$
\text{CPU} = \text{macchina sequeniale} \to \text{Stato}+\text{Circuito combinatorio}
$$
Per **stato** si intende un insieme di parametri che se inseriti nella macchina sequenziale i restituisce uno stesso output.

---
## Progettare la CPU MIPS
Prima fase: CPU MIPS semplice non ottimizzata (senza pipeline)
- definire come viene elaborata una istruzione (**fasi di esecuzione**)
- scegliere le **istruzioni da realizzare**
- scegliere le **unità fondamentali necessarie**
- **collegare** le unità funzionali
- **costruire la Control Unit** che controlla il funzionamento della CPU
	fa in modo che l’istruzione letta venga eseguita in modo corretto, sa quali parti della CPU attivare data una determinata sequenza binaria
- calcolare il massimo tempo di esecuzione delle istruzioni (che ci dà il **periodo del clock**)

Fasi di esecuzione di una istruzione:
- *fetch* → **caricamento** di una istruzione dalla memoria alla CU
- *decodifica* → **decodifica** della istruzione e **lettura argomenti** dai registri
- *esecuzione* → **esecuzione** (attivazione delle unità funzionali necessarie)
- *memoria* → accesso alla **memoria**
- *write back* → scrittura dei **risultati nei registri**
Altre operazioni necessarie:
- aggiornamento del PC (normale, salti condizionati, salti non condizionati)

Istruzioni da realizzare:
- accesso alla memoria → `lw, sw` (tipo I)
- salti condizionati → `beq` (tipo I)
- operazioni aritmetico-logiche → `add, sun, sll, slt` (tipo R)
- salti non condizionati → `j, jal` (tipo J)
- operandi non costanti → `li, addi, subi` (tipo I)