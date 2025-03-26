---
Created: 2025-03-26
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## Introduction
**Union-Find**, noto anche come *Disjoint Set Union* (DSU), è una struttura dati per gestire insiemi disgiunti. E’ utilizzato per operazioni di unione e ricerca efficienti su insiemi disgiunti

Le tre operazioni fondamentali di Union-Find sono:
1. $Crea(S)$ → restituisce una struttura dati Union-Find sull’insieme $S$ di elementi dove ciascun elemento è in un insieme separato
2. $Find(x,C)$ → restituisce il nome dell’insieme della struttura dati $C$ a cui appartiene l’elemento $x$ (a quele componente appartiene $x$)
3. $Union(A,B,C)$ → modifica la struttura dati $C$ fondendo la componente $A$ con la componente $B$ e restituisce il nome della nuova componente

Una gestione efficiente di insiemi disgiunti è utile in diversi contesti tra cui l’evoluzione di un grafo nel tempo attraverso l’aggiunta di archi. In questo caso gli insiemi disgiunti rappresentano le componenti connesse del grafo

Se viene aggiunto l’arco $(u,v)$, al grafo, si verifica innanzitutto se $u$ e $v$ sono nella stessa componente connessa. Se $A=find(u)$ e $B=find(v)$ risultano distinti allora l’operazione $Union(A,B)$ può essere usata per unire le due componenti

### Nome dell’insieme
Discutiamo brevemente su cosa intendiamo con **nome dell’insieme** (ad esempio quello ritornato dalla funzione $find$ su un elemento $x$). C’è un’ampia flessibilità nella scelta, come risulta dalle varie implementazioni la cosa importante è che $find(u)=find(v)$ se e solo se $u$ e $v$ sono nello stesso insieme.
La scelta fatta nelle implementazioni che seguono. è quella di scegliere come **nome dell’insieme quello di un particolare elemento dell’insieme stesso**
Come primo approccio possiamo pensare di assegnare all’insieme il nome dell’elemento massimo in esso contenuto

Probabilmente il modo più semplice di 