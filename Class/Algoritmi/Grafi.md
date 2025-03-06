---
Created: 2025-03-03
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#L’albero|L’albero]]
- [[#Grafo planare|Grafo planare]]
- [[#Rappresentazione di grafi tramite matrici binarie (matrice di adiacenza)|Rappresentazione di grafi tramite matrici binarie (matrice di adiacenza)]]
- [[#Rappresentazione di grafi tramite liste di adiacenza|Rappresentazione di grafi tramite liste di adiacenza]]
- [[#Esercizi|Esercizi]]
---
## Introduction

Un grafo è rappresentato da $G(V,E)$ dove $V$ è l’insieme dei vertici (nodi) e $E$ l’insieme degli archi
I grafi si distinguono in grafi **diretti** (gli archi sono con direzione) e **non diretti** (archi senza direzione).

Se il grafo è diretto il numero massimo di archi è $n(n-1)=O(n^2)$, mentre se è non diretto $m$ massimo è $\frac{n(n-1)}{2}=O(n^2)$

Un grafo si dice **sparso** se $m=O(n)$, e **denso** se $m=\Omega(n^2)$ (molti archi).
Inoltre un grafo denso si dice *completo* se ha tutti gli archi (es. $m=\theta(n^2)$), e *torneo* se tra ogni coppia di nodi c’è esattamente un arco (es. $m=\theta(n^2)$)

>[!warning]
>Un grafo non sparso non è necessariamente denso, ad esempio può avere $\theta(n\log n)$ archi

---
## L’albero
![[Pasted image 20250303111727.png|center|200]]
Un grafo **connesso** (esiste almeno un cammino tra ogni coppia di nodi) e senza **cicli** è detto **albero** (non necessariamente deve essere radicato)


>[!example] Grafo sconnesso con ciclo
>![[Pasted image 20250303111808.png|200]]

Un albero ha sempre $m=n-1$ archi, ed è facilmente dimostrabile per induzione. Si parla di **foglia** di un albero non radicato se un nodo ha un solo arco (ogni albero ha almeno due foglie). Per **grado** si intende il numero di archi che incidono sul nodo

---
## Grafo planare
I **grafi planari** sono quei grafi che posso disegnare sul piano senza che gli archi si intersechino

Nonostante possa non sembrare a primo impatto **tutti i grafi di 4 nodi sono planari** 
![[Pasted image 20250303112318.png|380]]

>[!example] Più piccolo grafo non planare
>![[Pasted image 20250303112424.png|200]]

>[!info] Gli alberi sono un sottoinsieme dei grafi planari
>![[Pasted image 20250303112539.png|400]]

Per il **teorema di Eulero** si può dire che un grafo planare di $n>2$ nodi ha al più $3n-6$ archi, dunque tutti i grafi planari sono sparsi
![[Pasted image 20250303112901.png|center|500]]
Dalla tabella si deduce che da $n=5$ in poi esistono di certo grafi non planari ($m$ in completi $>$ $m$ max in planari)

---
## Rappresentazione di grafi tramite matrici binarie (matrice di adiacenza)
Per rappresentare un grafo tramite matrice di adiacenza, metteremo $M[i][j]=1$ se e solo se c’è un arco diretto da $i$ a $j$

>[!example]
>![[Pasted image 20250303114309.png|200]]
>![[Pasted image 20250303114209.png|300]]
>>[!info] Se il grafo è non diretto, la matrice di adiacenza è sempre simmetrica rispetto alla diagonale

>[!example]
>![[Pasted image 20250303114617.png|200]]
>![[Pasted image 20250303114545.png|300]]

---
## Rappresentazione di grafi tramite liste di adiacenza
Utilizzo una lista di liste $G$, la lista ha tanti elementi quanti sono i nodi del grafo $G$. $G[x]$ è una lista contenente i nodi adiacenti al nodo $x$, ovvero i nodi raggiunti da archi che partono da $x$

>[!example]
>![[Pasted image 20250303114309.png|200]]
> ```python
> G=[
> 	[2, 5],       # vicini di 0
> 	[5],          # vicini di 1
> 	[0, 4, 5],    # vicini di 2
> 	[4],          # vicini di 3
> 	[2, 3, 5],    # vicini di 4
> 	[0, 1, 2, 4], # vicini di 5
> ]
> ```

>[!example]
>![[Pasted image 20250303114617.png|200]]
> ```python
> G=[
> 	[2, 5],    # vicini di 0
> 	[],        # vicini di 1
> 	[],        # vicini di 2
> 	[4],       # vicini di 3
> 	[2, 3, 5], # vicini di 4
> 	[0, 1],    # vicini di 5
> ]
> ```

>[!info] Vantaggi e svantaggi rispetto alla rappresentazione tramite matrice
>- notevole risparmio di spazio nel caso di grafi sparsi
>- vedere se due archi son connessi può costare anche $O(n)$

---
## Esercizi
![[Esercizi2#28/02#1]]

---
## Visitare in profondità (DFS)
### Matrice di adiacenza
Per visitare in profondità un grafo rappresentato tramite matrice di adiacenza:
```python
def DFSr(u, M, visitati):
	visitati[u] = 1
	for i in range(len(M)):                  # θ(n)
		# con il controllo su visitati evito di entrare in dei cicli
		# del grafo
		if M[u][i] == 1 and not visitati[i]:
			DFSr(i, M, visitati)             # O(n)

def DFS(u, M):
	visitati = [0]*len(M) # θ(n)
	DFSr(u, M, visitati)
	# in visitati[i] ho 1 se e solo se i è raggiungibile da u
	return [x for x in range(len(M)) if visitati[x]]
```

La complessità della procedura è $O(n)\times \theta(n)=O(n^2)$

### Liste di adiacenza
Per visitare in profondità un grafo rappresentato tramite liste di adiacenza:
```python
def DFSr(u, G, visitati):
	visitati[u] = 1
	for v in G[u]:
		if not visitati[v]:
			DFSr(v, G, visitati)

def DFS(u, G):
	visitati = [0]*len(G)
	DFSr(u, G, visitati)
	return [x for x in range(len(G)) if visitati[x]]
```

Ogni nodo viene visitato al più una volta e ogni elemento della lista contiene al più $n$ elementi; in questo caso quindi l’algoritmo controllerà gli adiacenti di tutti gli elementi, quindi il totale è  $\mid \text{adj 0}\mid+\mid \text{adj 1}\mid+\dots+\mid \text{adj }n-1\mid=m$ numero di archi

Si può quindi concludere che la complessità è $O(m+n)$ e la complessità di spazio è $O(n)$

>[!info] Se il grafo è sparso $m=O(n)$ quindi la DFS ha costo $O(n)$

>[!warning]
>Si sarebbe potuto risolvere questo problema e il precedente usando per visitati un insieme invece di una lista in modo tale da guadagnare a livello di spazio ma perdendo a livello temporale (per poter inserire un elemento si deve prima controllare se questo è già presente)

### Versione iterativa
```python
def DFS_iterativo(u, G):
	# visita dei nodi di G raggiunbili a partire dal nodo u
	visitati = [0]*len(G)
	pila = [u] # inizializza la pila con il nodo di partenza
	while pila:
		u = pila.pop()
		if not visitati[u]:
			visitati[u] = 1
			# aggiungiamo i vicini non visitati
			for v in G[u]:
				if not visitati[v]:
					pila.append(v)
	return [x for x in range(len(G)) if visitati[x]]
```

Al termine di `DFS_iterativo(u,G)` si ha `visitati[i]=1` se e solo se `i` è raggiungibile da `u`. La complessità di tempo della procedura è $O(n+m)$, mentre la complessità di spazio della procedura è $O(n)$

---
## Albero DFS
E’ possibile trasformare qualunque grafo in un albero; infatti con una visita DFS gli archi del grafo si bipartiscono in quelli che nel corso della visita sono stati attraversati (perché permettevano di raggiungere nuovi nodi) e gli altri.
I nodi visitati e gli archi effettivamente attraversati formano un albero detto **albero DFS**

>[!tldr] Procedimento
>Partendo da un grafo qualsiasi, scelgo il nodo da cui far partire la DFS (in questo caso è stato scelto il 9)
>![[Pasted image 20250306191115.png|350]]
>
>Quindi acceso tramite DFS ai vicini di 9 ordinati in ordine crescente (nella lista di adiacenza l’index 9 sarebbe del tipo `[0, 2, 8]`)
>![[Pasted image 20250306191738.png|350]]
>
>Adesso ho quindi un albero di questo tipo
>![[Pasted image 20250306191932.png|350]]

### Vettore dei padri
Un albero DFS può essere memorizzato tramite il **vettore dei padri**.
Il vettori dei padri `P` di un albero DFS di un grafo di $n$ nodi ha $n$ componenti in cui:
- se $i$ è un nodo dell’albero DFS → `P[i]` contiene il padre del nodo $i$ (per convenzione il padre della radice è la radice stessa)
- se $i$ non è un nodo dell’albero DFS → `P[i]` per convenzione contiene $-1$

