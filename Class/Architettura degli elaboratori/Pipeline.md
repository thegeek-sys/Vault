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
- Write Back (WB): **blocco registri

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

### Lettura e Scrittura dal blocco Registri
E’ però da notare che la fase di **lettura** (ID) che quella di **scrittura** (WB) lavorano sui registri, impiegando un tempo notevolmente minore (la metà in questo esempio, 100ps) rispetto a tutte le altre fasi dell’istruzione. Possiamo per questo eseguire ID e W nello stesso clock (periodo 200ps), suddividendolo in lettura e scrittura.
![[Screenshot 2024-05-02 alle 18.42.47.png]]

In particolar modo possiamo eseguire la **scrittura durante il rising edge** e la **lettura durante il falling edge** in modo tale da non perdere un intero ciclo di clock quando le due istruzioni si sovrappongono
![[Screenshot 2024-05-02 alle 18.44.10.png|450]]

---
## Criticità nell’esecuzione
L’implementazione della pipeline all’interno dell’architettura comporta anche la nascita di alcune criticità (**hazard**) dovute alla suddivisione in fasi delle istruzioni

Immaginiamo il caso in cui l’istruzione 1 **modifichi** il valore di un registro e l’istruzione 2 legga il valore di tale registro. Per via della suddivisione in fasi, durante la **fase di ID dell’istruzione 2** non è ancora stata eseguita la **fase di WB dell’istruzione 1**, generando quindi una **situazione critica** in cui il dato del registro non sia ancora stato modificato. Di conseguenza, l’istruzione 2 **leggerà il dato non ancora aggiornato**.

Gli hazard possono essere di tre tipi:
- **Structural hazard** → le risorse hardware non sono sufficienti (memoria dati e memoria istruzioni condivise in una singola memoria, risolto in fase di design)
- **Data hazard** → il dato necessario non è ancora pronto
- **Control hazard** → la presenza di un salto cambia il flusso di esecuzione delle istruzioni