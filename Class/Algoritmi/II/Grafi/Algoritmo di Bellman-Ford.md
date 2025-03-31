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
Questo garantisce che il costo minimo può essere calcolato considerando solo cammini di questa lunghezza, è quindi possibile considerare sottoproblemi che si ottengono limitando la lunghezza dei cammini.
![[Pasted image 20250328013014.png|600]]

Definiamo così la seguente tabella di dimensione $n\times n$
$$
T[i][j]=\text{costo di un cammino minimo da }s\text{ al nodo }j\text{ attraversando al più }i\text{ archi}
$$
Calcoleremo la soluzione al nostro problema determinando i valori della tabella. Infatti il costo minimo per andare da $s$ (sorgente) al generico nodo $t$ sarà $T[n-1][t]$

>[!example]
>Alla creazione, la tabella sarà del tipo:
>![[Pasted image 20250331105556.png]]
>
>Se $n-1$ e $n-2$ sono uguali allora ho il costo minimo, se sono diversi vuol dire che ci sta un ciclo negativo (i costi sono calcolati in base alla riga precedente)

I valori della prima riga della tabella $T$ sono ovviamente tutti $+\infty$ tranne $T[0][s]$ che vale $0$. Inoltre $T[i][s]=0$ per ogni $i>0$

Resta da definire la regola che permette di calcolare i valori delle celle $T[i][j]$ con $j\neq s$ della riga $i>0$ in funzione delle celle già calcolare della riga $i-1$
Distinguiamo due casi a seconda che il cammino di lunghezza al più $i$ da $s$ a $j$ abbia lunghezza esattamente $i$ o inferiore a $i$:
- nel primo caso ovviamente si ha $T[i][j]=T[i][j-1]$
- nel secondo caso deve invece esistere un cammino minimo di lunghezza al più $i-1$ ad un nodo $x$ e poi un arco 