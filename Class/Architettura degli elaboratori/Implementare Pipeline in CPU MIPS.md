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
![[FFA40860-4F15-464E-9035-C9ED6C4D2CA9_1_201_a 1.jpeg|550]]

Per questo motivo tutte le informazioni ed i segnali di controllo devono essere nel registro precedente della pipeline
![[Screenshot 2024-05-05 alle 19.06.38.png]]

---
## Con logica dei salti (beq)
Aggiungendo la logica dei salti (beq) e integrandola con i registri già esistenti, ne approfitto per spostare tutti i controlli dopo ID in modo tale da aver la necessità di effettuare controlli solo durante le ultime tre fasi dell’istruzione (rimane solamente `RegWrite` che però non mi crea problemi in quanto viene attivato solamente durante il WB).
Si noti che ora serve il campo `funz` (codice funzione) su 6 bit dell’istruzione, nello stadio EX dove viene utilizzato come ingresso del controllore della ALU; occorre quindi salvare anche questi bit nel registro di pipeline ID/EX
![[Screenshot 2024-05-06 alle 16.40.11.png]]