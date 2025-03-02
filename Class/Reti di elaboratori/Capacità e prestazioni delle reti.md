---
Created: 2025-03-02
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Introduction
L’internet può essere interpretato come una **rete di reti** composta da reti di accesso e backbone Internet

---
## Rete di acceso
![[Pasted image 20250302164539.png|center|350]]

La rete di accesso è composta da tutti i dispositivi che ci danno la possibilità di raggiungere il primo router del backbone (*edge router*); dove per **backbone** si intende la rete di tutti i router (composta solamente da router e dai loro collegamenti)

Il backbone Internet può essere di due tip (come già precedentemente detto):
- a **commutazione di circuito** → circuito dedicato per l’intera durata della sessione
- a **commutazione di pacchetto** → i messaggi di una sessione utilizzano le risorse su richiesta

---
## Struttura di Internet
La struttura di Internet è fondamentalmente gerarchica:
- ISP di primo livello → hanno la copertura più vasta, fornendo servizi a livello nazionale/internazionale. Un esempio è la **SEA-ME-WE 6**, il sistema South East Asia-Middle East-West Europe 6, che collega Singapore alla Francia attraverso cavi sottomarini e terrestri
- ISP di secondo livello → risulta essere ben più piccolo rispetto a quelli di primo livello provvedendo alla distribuzione nazionale o distrettuale, sfruttando i servizi disposti dagli ISP di primo livello
- ISP di terzo livello e ISP livello → chiamate *last hop network*, sono le reti più vicine ai sistemi terminali

![[Pasted image 20250302165735.png|center|550]]

Uno dei problemi principali delle reti è quello di trovare il percorso da seguire per arrivare a destinazione. Questo problema è denominato **routing** (instradamento)

---
## Capacità e prestazioni
Un argomento di cui si parla spesso nell’ambito delle reti è la velocità della rete o di un collegamento, ovvero quanto velocemente si riesce a trasmettere e ricevere i dati (seppur il concetto di velocità sia più ampio e coinvolga più fattori).

Nel caso di una rete a commutazione di pacchetto, le metriche ne determinano le prestazioni e si misurano in:
- Ampiezza di banda → bandwidth e bit rate
- Throughput
- Latenza
- Perdita di pacchetti

---
## Bandwidth e bit rate
Con il termine **ampiezza di banda** si indicano due concetti diversi ma strettamente legati: la **bandwidth** e il **bit rate**

La **bandwidth** è una caratteristica del mezzo trasmissivo, si misura i Hz e rappresenta l’**ampiezza dell’intervallo di frequenza utilizzato dal mezzo trasmissivo**.
Maggiore e la ampiezza di banda maggiore è la quantità di informazione che può essere veicolata attraverso il mezzo.

Il **bit rate** indica quanti bit posso trasmettere per unità di tempo; la sua unità di misura è il bps (bit per second)

Il bit rate dipende sia dalla banda (maggiore è la banda maggiore è il bit rate) che dalla specifica tecnica di trasmissione

---
## Throughput
Throughput è una misura che ci dice quanto **effettivamente** (in contrapposizione a nominalmente) una rete riesce ad inviare dati. Somiglia al bit rate ma tendenzialmente gli è minore o uguale (il bit rate è la potenziale velocità di un link)

>[!example]
>Una strada è progettata per far transitare $1000$ auto al minuto da un punto all’altro. Se c’è traffico, tale cifra può essere ridotta a $100$. Il rate è $1000$ auto al minuti, il troughput è di $100$ auto al minuto

