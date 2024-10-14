---
Created: 2024-10-14
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
Un sistema operativo deve allocare risorse tra diversi processi che ne fanno richiesta contemporaneamente. Tra le diverse possibili risorse, c’è il tempo di esecuzione, che viene fornito da un processore. Questa risorsa viene allocata tramite lo **scheduling**

### Scopo dello scheduling
Dunque lo scopo dello scheduling è quello di assegnare ad ogni processore i processi da eseguire, man mano che i processi stessi vengono creati e distrutti. Tale obiettivo va raggiunto ottimizzando vari aspetti:
- tempo di risposta
- throughput
- efficienza del processore

### Obiettivi dello scheduling
L’obiettivo è dunque quelli di distribuire il tempo di esecuzione in modo equo tra i vari processi, ma al tempo stesso anche gestire le priorità dei processi quando necessario (es. vincoli di real time ovvero eseguire operazioni entro un certo tempo).
Deve inoltre evitare la starvation dei processi, ma anche avere un **overhead** basso, ovvero avere un tempo di esecuzione dello scheduler stesso basso

---
## Tipi di Scheduling
Esistono vari tipi di scheduler in base a quanto spesso un processo viene eseguito:
- *Long-term scheduling* → decide l’aggiunta ai processi da essere eseguiti (eseguito molto di rado)
- *Medium-term scheduling* → decide l’aggiunta ai processi che sono in memoria principale (eseguito sempre non molto spesso)
- *Short-term scheduling* → decide quale processo, tra quelli pronti, va eseguito da un processore(eseguito molto spesso)
- *I/O scheduling* → decide a quale processo, tra quelli con una richiesta pendente per l’I/O, va assegnato il corrispondente dispositivo di I/O

---
## Processi e scheduling
### Stati dei processi
Ricordando il modello a 7 stati, le transizioni tra i vari stati sono decise dallo scheduler
![[Pasted image 20241014213242.png|500]]
Il long-term scheduler si occupa della creazione dei nuovi processi
Il mid-term scheduler si occupa dello swapping tra memoria principale e disco (e viceversa)
Lo short-term scheduler si occupa di decidere quali processi ready devono essere running

>[!info]- Modello a 7 stati
>![[Processi#Processi sospesi]]

### Code dei processi
![[Pasted image 20241014213625.png|550]]

---
## Long-term scheduling

