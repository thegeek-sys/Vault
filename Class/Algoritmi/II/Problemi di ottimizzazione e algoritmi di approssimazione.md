---
Created: 2025-04-05
Class: "[[Algoritmi]]"
Related:
---
---
## Introduction
Un problema di ottimizzazione è un tipo di problema in cui l’obbiettivo è trovare la migliore soluzione possibile tra un insieme di soluzioni ammissibili.
Ogni soluzione ammissibile, cioè una soluzione che soddisfa tutte le condizioni imposte dal problema, ha un valore associato che può essere un “costo” o un “beneficio”

A seconda del tipo di problema, l’obiettivo può essere minimizzare questo valore (es. ridurre i costi) o massimizzarlo (es. ottenere il massimo guadagno). 
Abbiamo quindi problemi di **minimizzazione** e problemi di **massimizzazione**

>[!example]
>Consideriamo il problema dello spanning tree (albero di copertura minimo) su un grafo pesato.
>Una soluzione ammissibile in questo caso è un albero di copertura, ovvero un sottoinsieme di archi che connette tutti i nodi del grafo senza formare cicli. La soluzione ottima è l’albero di copertura che ha il costo minimo, cioè la somma dei pesi degli archi è la più bassa possibile tra tutti gli alberi di copertura. In questo caso, trovare una soluzione ottima è un problema che può essere risolto in tempo polinomiale (es. Kruskal)
>
>Un altro esempio di problema di ottimizzazione è la ricerca del cammino minimo tra due nodi $a$ e $b$ in un grafo pesato.
>Una soluzione ammissibile è un cammino che collega $a$ e $b$ attraverso una sequenza di archi del grafo. La soluzione ottima, invece, è il cammino che ha il costo minimo, cioè la somma dei pesi degli archi del cammino è la più bassa possibile. Anche in questo caso trovare una soluzione ottima è un problema che può essere risolto in tempo polinomiale (es. Dijsktra)

