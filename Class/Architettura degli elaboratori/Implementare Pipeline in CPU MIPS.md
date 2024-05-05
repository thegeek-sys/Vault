---
Created: 2024-05-05
Class: "[[Architettura degli elaboratori]]"
Related:
  - "[[Pipeline]]"
Completed:
---
---
## Introduction
Adesso che è chiaro cosa sia una pipeline possiamo iniziare a pensare a come implementarla all’interno di una CPU MIPS.
Per farlo e permettere quindi il **forwarding** (operazione cardine all’interno della pipeline) è necessario integrare dei registri tra le varie unità funzionali, in modo tale da poter recuperare un dato se necessario
![[registri pipeline_1.jpeg]]
### Esempio
Prendiamo in analisi il seguente codice:
![[FFA40860-4F15-464E-9035-C9ED6C4D2CA9_1_201_a.jpeg]]

In questo caso avremmo un problema durante il Write Back dell’istruzione `lw`. Questo avviene in quando se guardiamo le fasi dell’esecuzione risulta facile notare come, nel momento in cui l’istruzione $\enclose{circle}{1}$ esegue il WB l’istruzione $\enclose{circle}{4}$ ha già eseguito IF e dunque all’interno del blocco dei registri sono già pronti i registri del `sw` per essere letti e scritti. Dunque risulterebbe che il registro di destinazione di `lw` al posto di essere `$t4` risulta essere `$t6`
