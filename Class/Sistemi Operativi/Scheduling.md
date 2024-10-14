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