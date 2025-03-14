---
Created: 2025-03-07
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Index
- [[#Visitare in profondità (DFS)|Visitare in profondità (DFS)]]
	- [[#Visitare in profondità (DFS)#Matrice di adiacenza|Matrice di adiacenza]]
	- [[#Visitare in profondità (DFS)#Liste di adiacenza|Liste di adiacenza]]
	- [[#Visitare in profondità (DFS)#Versione iterativa|Versione iterativa]]
- [[#Albero DFS|Albero DFS]]
	- [[#Albero DFS#Vettore dei padri|Vettore dei padri]]
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

La complessità della procedura è $O(n)\times \Theta(n)=O(n^2)$

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
	return [x for x i range(len(G)) if visitati[x]]
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

>[!example] Esempio albero precedente
>![[Pasted image 20250306192823.png|400]]

Modificando leggermente la procedura DFS  è possibile fare in modo che restituisca il vettore dei padri `P` anziché il vettore dei visitati

```python
def DFSr(x, G, P):
	for y in G[x]:
		if P[y] == -1:
			P[y] = x
			DFSr(y, G, P)

def Padri(u, G):
	n = len(G)
	P = [-1]*n
	P[u] = [u]
	DFSr(u, G, P)
	return P

# >> G=[[1], [2,3,5], [4], [5], [6], [], [2]]
# >> Padri(0,G)
# [0, 0, 1, 1, 2, 3, 4]
```

In molti casi non basta sapere se un nodo $y$ è raggiungibile a partire dal nodo $x$ de grafo, si vuole anche sapere il **cammino che lo permette**. Il vettore dei padri dell’albero DFS, radicato in $x$, permette di ricavare facilmente tale cammino.
Basta controllare che il nodo $y$ sia nell’albero e poi da $y$ risalire ed effettuare `reverse` dei nodi incontrati

>[!example] Nodo dei padri precedente
>Immaginiamo di voler sapere il cammino che da 9 porta a 7
>![[Pasted image 20250306193657.png|400]]
>
>Invertendo si avrebbe `[9, 2, 7]`

Procedura *iterativa* per la ricerca del cammino:
```python
def Cammino(u, P):
	path = []
	if P[u] == -1: return path
	while P[u] != u:
		path.append(u)
		u = P[u]
	path.append(u)
	path.reverse()
	return path
```
Disponendo del vettore dei padri, la complessità è $O(n)$

Procedura *ricorsiva* per la ricerca del cammino:
```python
def CamminoR(u, P):
	if P[u] == -1: return []
	if P[u] == u: return [u]
	return CamminoR(P[u], P) + [u]
```
Disponendo del vettore dei padri, la complessità è $O(n)$

>[!warning]
>Se esistono più cammini che dal nodo $x$ portano al nodo $y$ la procedura appena vista non garantisce di restituire il **cammino minimo** (vale a dire quello che attraversa il minor numero di archi)
>![[Pasted image 20250306195630.png|400]]
>
>Il cammino minimo da $4$ a $3$ è `[4, 3]` mentre la procedura restituisce il cammino `[4, 0, 1, 2, 3]`

