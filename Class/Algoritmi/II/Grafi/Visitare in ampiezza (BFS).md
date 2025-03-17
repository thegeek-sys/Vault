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

Per effettuare questo tipo di visita manteniamo in una **coda** i nodi visitati i cui adiacenti non sono stati ancora esaminati. Ad ogni passo, preleviamo il primo nodo dalla coda, esaminiamo i suoi adiacenti e se scopriamo un nuovo nodo lo visitiamo e lo aggiungiamo alla coda.

![[Pasted image 20250314112303.png|500]]

I relativi **alberi dei cammini minimi** risultati da tre visite BFS del grafo $G$ che partono da $0$, $5$ e $2$ rispettivamente
![[Pasted image 20250314112347.png|500]]

Si comincia con una coda contenente il solo nodo di partenza x. Fino a che la coda non risulta vuota: ad ogni passo un nodo viene estratto dalla coda, tutti i suoi adiacenti vengono visitati e messi in coda
```python
def BFS(x, G):
	visitati = [0]*len(G)       # θ(n)
	visitati[x] = 1
	coda = [x]
	while coda:
		u = coda.pop(0)          # O(n)
		for y in G[u]:           # O(n)
			if visitati[y] == 0: # O(m)
				visitati[y] = 1
				coda.append(y)
	return visitati

# >> G=[[1,5], [2], [3], [4], [], [2,4], [2]]
# >> BFS(0,G)
# [1,1,1,1,1,1,0]
```
Alla fine $\text{visitati}[u]=1$ se e solo se $u$ è raggiungibile da $x$.
Un nodo finisce in coda al più una volta (poi risulterà visitato e non ci finisce più) quindi il `while` verrà eseguito $O(n)$ volte. Le liste di adiacenza dei nodi verranno scorse al più una volta quindi il costo totale del `for` sarò $O(m)$. Infine l’estrazione in testa `pop(0)` ha costo $O(n)$.
Inoltre poiché vengono sempre visitati archi e nodi diversi posso sommare $n$ e $m$ (per lo stesso motivo della [[Visitare in profondità (DFS)#Visitare in profondità (DFS)#Liste di adiacenza|DFS]]). Quindi la complessità è:
$$
\theta (n)+O(m)+O(n^2)=O(n^2)
$$

### Ottimizzazione
Se però riuscissimo a eseguire `pop(0)` in $O(1)$ il costo diventerebbe $O(n+m)$
Per farlo mi è sufficiente effettuare solo **cancellazioni logiche** e non effettive. Uso un puntatore $i$ che indica l’inizio della coda all’interno della lista. Il puntatore si incrementa ogni volta che si cancella nella coda (quindi il test di coda vuota va opportunamente modificato)
```python
def BFS(x, G):
	visitati = [0]*len(G)       # θ(n)
	visitati[x] = 1
	coda = [x]
	i = 0
	while len(coda) > i:
		u = coda[i]              # O(1)
		i += 1
		for y in G[u]:           # O(n)
			if visitati[y] == 0: # O(m)
				visitati[y] = 1
				coda.append(y)
	return visitati
```
Ora la procedura ha complessità $O(n+m)$

---
## BFS e albero dei padri
Modifichiamo ora la procedura in modo tale che restituisca in $O(n+m)$ l’albero di visita BFS rappresentato tramite vettore dei padri

>[!info]
>Grazie al vettore dei padri $P$ con la procedura $\text{cammini}(x,P)$ in $O(n)$ potremo ottenere un cammino in $G$ dalla radice dell’albero al nodo $x$ (posto che $x$ sia raggiungibile)

```python
def BFSpadri(x, G):
	P = [-1]*len(G)
	P[x] = x
	coda = [x]
	i = 0
	while len(coda) > i:
		u = coda[i]
		i += 1
		for y in G[u]:
			if P[y] == -1:
				P[y] = u
				coda.append(y)
	return P
```

>[!info] Proprietà
>La distanza minima di un vertice $x$ da $s$ nel grafo $G$ equivale alla profondità di $x$ nell’albero BFS
>>[!done] Dimostrazione
>>Per induzione sulla distanza $d$ di $x$ da $s$. E’ ovviamente vero per $d=0$ in quando l’univo vertice a distanza $0$ da $s$ è $s$ stesso (che è a profondità $0$ nell’albero).
>>Supponiamo sia vero per tutti i vertici con distanza al più $d-1$ e consideriamo quindi un vertice $x$ a distanza $d$. Sia $P$ un cammino da $s$ a $x$ e sia $v$ il predecessore di $x$ in questo cammino. Per ipotesi induttiva $v$ è a profondità $d-1$ nell’albero BFS.
>>Se $x$ è stato inserito nell’albero grazie a $v$ allora si troverà a profondità $d$, assumiamo che $v$ sia stato inserito nell’albero grazie ad un nodo $u\neq v$
>>La profondità di $u$ non può essere inferiore a $d-1$ altrimenti avremmo trovato un cammino che parte da $s$ e porta a $v$ (tramite $u$) di lunghezza inferiore a $d$. D’altra parte non può essere la profondità di $u$ maggiore di $d-1$ perché il nodo $v$ sarebbe stato visitato prima di $u$ e $x$ sarebbe stato inserito grazie al nodo $v$. Deve quindi aversi che la profondità di $u$ è $d-1$ quindi la profondità di $v$ è comunque $d$

Grazie alla proprietà appena dimostrata i cammini prodotti grazie all’albero sono quelli di lunghezza minima ecco perché l’albero BFS che si ottiene dalla visita è detto anche albero dei cammini minimi

---
## BFS e vettore delle distanze $D$
Modifichiamo ora leggermente la procedura di visita in modo che restituisca in $O(n+m)$ il vettore delle distanze $D$
Al nodo $x$ viene assegnata distanza zero e a tutti gli altri nodi il valore $-1$. A ciascun nodo via via visitato viene assegnata la distanza corrispondente al padre incrementata di $1$. 
Al termine $D[u]$ conterrà $-1$ se il nodo $u$ non è raggiungibile a partire da $x$, la distanza minima di $u$ da $x$ altrimenti

```python
def BFSdistanze(x, G):
	D = [-1]*len(G)
	D[x] = 0
	coda = [x]
	i = 0
	while  len(coda) > i:
		u = coda[i]
		i += 1
		for y in G[u]:
			if D[y] == -1:
				D[y] = D[u]+1
				coda.append(y)
	return D
```

