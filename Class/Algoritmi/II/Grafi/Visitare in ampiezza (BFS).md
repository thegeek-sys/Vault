---
Created: 2025-03-14
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction
Dati due nodi $a$ e $b$ di un grafo $G$, definiamo **distanza** (minima) in $G$ di $a$ da $b$ il numero minimo di archi che bisogna attraversare per raggiungere $b$ a partire da $a$. La distanza è posta a $+\infty$ se $b$ non è raggiungibile partendo da $a$, e $0$ se è il nodo da cui sto calcolando le distanze (dista $0$ da sé stesso)

![[Pasted image 20250314111202.png|center|400]]
In figura sono riportate accento ad ogni nodo le distanze dal nodo $2$ in rosso

>[!question] Problema
>Dato un grafo $G$ ed un suo nodo $x$ vogliamo conoscere le distanze di tutti i nodi di $G$ da  $x$
>Più precisamente vogliamo calcolare il vettore delle distanze $D$, dove in $D[y]$ troviamo la distanza di $y$ da $x$
>![[Pasted image 20250314111528.png]]

---
## Visita in ampiezza
La **visita in ampiezza** (*Breadth First Search*) esplora i nodi del grafo partendo da quelli a distanza $1$ dalla sorgente $s$. Poi visita quelli a distanza $2$ e così via. L’algoritmo visita tutti i vertici a livello $k$ prima di passare a quelli a livello $k+1$

![[Pasted image 20250314111723.png|center|650]]

Si genera così un albero detto **albero BFS** tramite il quale si visitano nodi sempre più distanti dalla radice

Per effettuare questo tipo di visita manteniamo in una coda i nodi visitati i cui adiacenti non sono stati ancora esaminati. Ad ogni passo, preleviamo il primo nodo dalla coda, esaminiamo i suoi adiacenti e se scopriamo un nuovo nodo lo visitiamo e lo aggiungiamo alla coda.

