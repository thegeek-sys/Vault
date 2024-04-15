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

> [!hint]
> La cosa particolare sta nel fatto che i passaggi da eseguire per una singola istruzione li posso fare in parallelo. Se ad esempio faccio un `add $s1,$t0,$t1` posso contemporaneamente leggere i valori di $t1 e $t0 e poi in un secondo momento fare la somma

Istruzioni da realizzare:
- **accesso alla memoria** → `lw, sw` (tipo I)
- **salti condizionati** → `beq` (tipo I)
- **operazioni aritmetico-logiche** → `add, sun, sll, slt` (tipo R)
- **salti non condizionati** → `j, jal` (tipo J)
- **operandi non costanti** → `li, addi, subi` (tipo I)

Formato delle istruzioni MIPS (ovvero la loro codifica in memoria)
![[Screenshot 2024-04-15 alle 16.09.41.png]]

---
## Unità funzionali necessarie
- **PC** → registro che contiene l’indirizzo della istruzione
- **memoria istruzioni** → contiene le istruzioni
- **memoria dati** → da cui leggere/in cui scrivere i dati (load/store)
- **adder** → per calcolare il PC (successivo o salto)
- **registri** → contengono gli argomenti delle istruzioni
- **ALU** → fa le operazioni aritmetico-logiche, confronti, indirizzi in memoria

Le varie unità funzionali sono collegate tra loro attraverso diversi **datapath** (interconnessioni che definiscono il flusso delle informazioni nella CPU, i cavi)

Se un’unità funzionale può ricevere dati da **più sorgenti** è necessario inserire un multiplexer (**MUX**) per selezionare la sorgente necessaria

Le unità funzionali sono attivate e coordinate dai segnali prodotti dalla **Control Unit**

