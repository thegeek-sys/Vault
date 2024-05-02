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

Queste cinque fasi vengono però svolte in sequenza poiché non è possibile eseguire la fase successiva senza il risultato della precedente, rendendo quindi le altre 4 fasi temporaneamente inutilizzabili.

Per rendere il più efficiente possibile la nostra architettura, dunque, possiamo scomporre l’esecuzione di un’istruzione in una catena di montaggio (**Pipeline**), dove ogni fase svolge il compito ad essa assegnatogli per poi **passare il risultato alla fase successiva**, procedendo ad elaborare già la **stessa fase** dell’istruzione successiva:
![[Screenshot 2024-05-02 alle 16.41.46.png]]

---
## Incremento della velocità
Utilizzando la pipeline possiamo quindi **ridurre il periodo di clock**, dalla durata massima di un’istruzione, alla **durata massima di una fase** (se fosse minore si accumulerebbero le istruzioni e dunque non potrebbero essere lette ed eseguite in modo corretto).
Nel caso ideale in cui ogni fase impiega lo stesso tempo, dunque la velocità sarà cinque volte più veloce rispetto all’esecuzione sequenziale delle fasi.

Immaginiamo che le 5 fasi abbiano le seguenti tempistiche:
- *Instruction Fetch* → 200ps  
- *Instruction Decode* → 100ps
- *Instruction Execute* → 200ps
- *Memory Access* → 200ps
- *Write Back* → 100ps
Normalmente, per poter eseguire l’istruzione più lenta possibile, ossia richiedente il completamento di tutte e 5 le fasi (ad esempio Load Word), sarebbe necessario utilizzare un periodo di clock pari a 800ps.
Tramite l’implementazione della pipeline, invece, tale periodo può essere ridotto a quello della **fase più lenta**, ossia $200\text{ps}$, aumentando quindi la velocità dell’architettura.