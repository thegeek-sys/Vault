---
Created: 2024-05-02
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Fino ad ora abbiamo immaginato ogni istruzione come divisa in cinque fasi:
- Instruction fetch (IF): **memoria istruzioni (e aggiornamento PC)**
- Instruction Decode (ID): **blocco registri (e CU)**
- Execute (EXE): **ALU**
- Memory access (MEM): **memoria dati**
- Write Back (WB): **banco registri

Queste cinque fasi vengono però svolte in sequenza poiché non è possibile eseguire la fase successiva senza il risultato della precedente, rendendo quindi le altre 4 fasi temporaneamente inutilizzabili
Per rendere il più efficiente possibile la nostra architettura, dunque, possiamo scomporre l’esecuzione di un’istruzione in una catena di montaggio (**Pipeline**), dove ogni fase svolge il compito ad essa assegnatogli per poi **passare il risultato alla fase successiva**, procedendo ad elaborare già la **stessa fase** dell’istruzione successiva:
