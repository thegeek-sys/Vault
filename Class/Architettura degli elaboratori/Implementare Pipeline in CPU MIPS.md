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
Prendiamo in analisi il seguente codice.
![[DB58386A-20B1-451C-81EC-CA9DEA05AED4_1_201_a.jpeg]]

In questo caso avremmo un problema durante il Write Back dell’istruzione `lw`. Questo avviene in quando se guardiamo le fasi dell’esecuzione risulta facile notare come, nel momento in cui l’istruzione $\raisebox{.5pt}{\textcircled{\raisebox{-.9pt} {8}}}$