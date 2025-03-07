---
Created: 2025-03-07
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction
Dato un grafo connesso $G$ ed un intero $k$ vogliamo sapere se è possibile colorare i nodi del grafo in modo che i nodi adiacenti abbiano sempre colori distinti

>[!example] Esempio di grafo 3-colorabile
>![[Pasted image 20250307103527.png|550]]

---
## Teorema dei 4 colori

>[!info] Teorema dei 4 colori
>Un grafo **planare** richiede al più 4 colori per essere colorato

Il problema venne posto per la prima volta nel 1852 da uno studente che congetturò che 4 colori sono sempre sufficienti. Negli anni successivi molti matematici tentarono invano di dimostrare la congettura.
La prima dimostrazione fu proposta solo nel 1879, ma nel 1890 si scoprì che la dimostrazione conteneva un sottile errore. Si provò almeno che 5 colori sono sempre sufficienti a colorare una mappa (tramite un’induzione).
La dimostrazione che 4 colori sono sufficienti fu trovata solo nel 1977. Si basa sulla riduzione del numero infinito di mappe possibili a 1936 configurazioni, per le quali la validità del teorema viene verificata caso per caso con l’ausilio di un calcolatore
Nel 2000 infine è stata proposta una nuova dimostrazione del teorema che richiede l’utilizzo della teoria dei gruppi

In generale si può dire che un grafo può richiedere anche $\theta(n)$ colori. Inoltre non è noto alcun algoritmo polinomiale che, dato un grafo planare $G$, determini se $G$ è 3-colorabile, ma **non è difficile progettare un algoritmo che determini se un grafo è 2-colorabile**

---
## Grafi 2-colorabili

>[!info] Un grafo è 2-colorabile se e solo se **non** contiene cicli di lunghezza dispari

Infatti un ciclo di lunghezza dispari rende impossibile la colorazione del grafo di due colori:
![[Pasted image 20250307104502.png]]

### Algoritmo
L’algoritmo di bi-colorazione che segue prova che un grafo senza cicli dispari può essere sempre 2-colorato:
- colora il nodo $0$ con il colore $0$
- effettua una visita in profondità del grafo a partire dal nodo $0$. Nel corso della visita, a ciascun nodo $x$ che incontri assegna uno dei colori $0$ e $1$. Scegli il colore da assegnare in modo che sia diverso dal colore assegnato al nodo padre che ti ha portato a visitare $x$

>[!info] Dimostrazione
>Siano $x$ e $y$ due nodi adiacenti in $G$, consideriamo i due possibili casi e facciamo vedere che in entrambi i casi i due nodi al termine avranno colori opposti
>1. L’arco $(x,y)$ viene attraversato durante la visita → in questo caso banalmente hanno colori distinti
>2. L’arco $(x,y)$ non viene attraversato durante la visita → sia $x$ il nodo visitato prima. Esiste un cammino in $G$ che porta da $x$ porta a $y$ (quello seguito dalla visita), questo cammino si chiude a formare un ciclo con l’arco $(y,x)$. Il ciclo è di lunghezza pari per ipotesi, quindi il cammino è di lunghezza dispari. Poiché sul cammino i colori si alternano il primo nodo ($x$) e l’ultimo nodo ($y$) del cammino avranno colori diversi

```python
def DFSr(x, G, Colore, c):
	Colore[x] = c
	for y in G[x]:
		if Colore[y]==-1:
			DFSr(y, G, Colore, 1-c)

def Colora(G):
	Colore = [-1]*len(G)
	DFSr(0, G, Colore, 0)
	return Colore
```

>[!warning] Se il grafo $G$ contiene cicli dispari l’algoritmo produce un assegnamento di colori sbagliato

Nella versione che segue l’algoritmo produce una bi-colorazione se il grafo $G$ è bi-colorabile, produce una lista vuota in caso contrario:
```python
def DFSr(x, G, Colore, c):
	Colore[x] = c
	for y in G[x]:
		if Colore[y]==-1:
			if not DFSr(y, G, Colore, 1-c):
				return False
		elif Colore[y] == Colore[x]:
			return False
	return True

def Colora1(G):
	Colore = [-1]*len(G)
	if DFSr(0, G, Colore, 0):
		return Colore
	return []
```
La complessità dell’algoritmo per testare se un grafo è bicolorabile è quella di una semplice visita del grafo connesso da colorare:
$$
O(n+m)=O(m)
$$
Dove l’ultima uguaglianza dipende dal fatto che in un grafo connesso $m\geq n-1$

---
## Componente connessa
Una **componente connessa** du un grafo (indiretto) è un sottografo composto da un insieme massimale di nodi connessi da cammini. Un grafo si dice connesso se ha una sola componente
![[Pasted image 20250307114201.png]]
Vogliamo calcolare il **vettore $C$ delle componenti connesse** di un grafo $G$. Vale a dire il vettore `C` che ha tanti elementi quanti sono i nodi del grafo e `C[u]=C[v]` se e solo se `u` e `v` sono nella stessa componente connessa
![[Pasted image 20250307114326.png]]

```python
def DFSr(x, G, C, c):
	C[x] = c
	for y in G[x]:
		if C[y] == 0:
			DFSr(y, G, C, c)

def Componenti(G):
	C = [0]*len(G)
	c = 0
	for x in range(len(G)):
		if C[x] == 0:
			c+=1
			DFSr(x, G, C, c)
	return C

# >> G = [[1, 5], [0, 5], [4], [], [2], [0, 1]]
# >> Componenti(G)
# [1, 1, 2, 3, 2, 1]
```
Poiché ogni visita agisce su archi diversi, il costo rimane $O(n+m)$. Se ad esempio abbiamo 3 componenti connesse si ha:
$$
O(n_{1}+m_{1})+O(n_{2}+m_{2})+O(n_{3}+m_{3})=O(n+m)
$$
Infatti la somma deve essere la totalità dei nodi

---
## Componente fortemente connessa
