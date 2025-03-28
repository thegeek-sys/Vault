---
Created: 2025-03-28
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction

>[!question] Problema
>Dato un grafo diretto e pesato $G$ in cui i pesi degli archi possono essere anche negativi e fissato un suo nodo $s$, vogliamo determinare il costo minimo dei cammini che conducono da $s$ a tutti gli altri nodi del grafo. Se non esiste un cammino verso un determinato nodo il costo sarà infinito

In questo primo problema però non è considerata la possibilità di poter avere un **ciclo negativo**, ovvero un ciclo diretto in un grafo in cui la somma dei pesi degli archi che lo compongono è negativa

>[!example]
>![[Pasted image 20250328011833.png|350]]
>Il ciclo evidenziato in figura è negativo di costo $5-3-6+4-2=\textcolor{red}{-2}$

Infatti se in un cammino tra i nodi $s$ e $t$ è presente un nodo che appartiene ad un ciclo negativo allora non esiste un cammino minimo tra $s$ e $t$ (è $-\infty$)
![[Pasted image 20250328012200.png|400]]

Se per il ciclo $W$ si ha costo $\text{costo}(W)<0$, ripassando più volte attraverso il ciclo $W$ possiamo arbitrariamente abbassare il costo del cammino da $s$ a $t$

Alla luce di quanto appena detto sui cicli negativi, la formulazione del problema non era del tutto corretta. Ecco di seguito la versione corretta

>[!question] Problema
>Dato un grafo diretto e pesato $G$ in cui i pesi degli archi possono essere anche negativi **ma che non contiene cicli negativi**, e fissato un suo nodo $s$, vogliamo determinare il costo minimo dei cammini che conducono da $s$ a tutti gli altri nodi del grafo. Se non esiste un cammino verso un determinato nodo il costo sarà infinito

Per risolvere questo problema, come abbiamo già visto, non è possibile usare l’algoritmo di Dijsktra. Useremo quindi l’**algoritmo di Bellman-Ford**, di complessità $O(n^2+m\cdot n)$

>[!info] Proprietà
>Se il grafo $G$ non contiene cicli negativi, allora per ogni nodo $t$ raggiungibile dalla sorgente $s$ esiste un cammino di costo minimo che attraversa al più $n-1$ archi

Infatti, se un cammino avesse più di $n-1$ archi, allora almeno un nodo verrebbe ripetuto, formando un ciclo. Poiché il grafo non ha cicli negativi, rimuovere eventuali cicli dal cammino **non aumenta** il suo costo complessivo, di conseguenza esiste sempre un cammino ottimale di lunghezza $n-1$
Questo garantisce che il costo minimo può essere calcolato considerando solo cammini di questa lunghezza
![[Pasted image 20250328013014.png|600]]
