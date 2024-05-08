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

---
## Segnali di controllo della CU
Possiamo quindi dividere i segnali di controllo nelle fasi esecutive in 3 gruppi:
- fase EXE
- fase MEM
- fase WB
![[Screenshot 2024-05-06 alle 16.50.34.png]]

Così si propagano i segnali di controllo all’interno della CPU
![[Screenshot 2024-05-06 alle 16.56.42.png]]

CPU nella sua totalità completa di segnali di controllo uscenti dalla porzione dei registri di pipeline dedicata a memorizzare le informazioni di controllo
![[Screenshot 2024-05-06 alle 16.58.29.png]]

---
## Scoprire un data hazard in EXE
Immaginiamo di avere questa sequenza di istruzioni:
![[Screenshot 2024-05-08 alle 16.38.51.png]]

In questo caso, nonostante tutte le istruzioni utilizzino il registro `$2` le uniche istruzioni che avranno il risultato corretto di `sub` saranno `add` e `sw` in quanto le altre due leggerebbero solamente il valore precedentemente immagazinato in `$2`.

Ma risulta facile capire come il risultato dell’istruzione `sub` sia disponibile già al termine della fase EX.
Ma quando hanno realmente bisogno l’istruzione `and` e `or` del dato? Hanno bisogno del dato di `$2` solamente all’inizio della fase EX (cc 4 e 5), di conseguenza è possibile eseguire questo frammento di codice senza stalli semplicemente **propagando** il dato a qualsiasi unità lo richieda non appena è disponibile.

Questo veloce esempio ci fa subito capire quali siano le casistiche che ci portano ad un data hazard in EXE:
1. $\text{EX/MEM.RegistroRd}=\text{ID/EX.RegistroRs}$
2. $\text{EX/MEM.RegistroRd}=\text{ID/EX.RegistroRt}$
3. $\text{MEM/WB.RegistroRd}=\text{ID/EX.RegistroRs}$
4. $\text{MEM/WB.RegistroRd}=\text{ID/EX.RegistroRt}$

Il primo hazard generato dall’esempio è quello tra `sub $2,$1,$3` e `and $12,$2,$5`. Tale hazard può essere rilevato quando l’istruzione `and` si trova allo stadio EX e l’istruzione precedente si trova nello stadio MEM; si tratta quindi di un hazard di tipo 1 $\text{EX/MEM.RegistroRd}=\text{ID/EX.RegistroRs}=\$2$

Ma dato che **non tutte le istruzioni scrivono il risultato nel register file**, questa strategia non è precise e potrebbero esserci casi in cui viene propagato un dato anche se non è necessario.
Una possibile soluzione consiste nel verificare se il segnale **`RegWrite` è attivo**