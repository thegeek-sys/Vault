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
