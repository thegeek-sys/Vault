---
Created: 2025-03-08
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#DAG|DAG]]
	- [[#DAG#Algoritmo alternativo|Algoritmo alternativo]]
---
## Introduction
Spesso un grafo diretto cattura relazione di *propedeuticità* (una arco da $a$ a $b$ indica che $a$ è propedeutico a $b$).
Potrò rispettare tutte le propedeuticità se riesco ad ordinare i nodi del grafo in modo che gli archi vadano tutti da sinistra verso destra. Questo ordinamento è detto **ordinamento topologico**

![[Pasted image 20250308161735.png]]

Un grafo diretto può avere da $0$ a $n!$ ordinamenti topologici. Un algoritmo esaustivo per il problema ha complessità $\Omega(n!)$

---
## DAG
Perché un grafo $G$ possa avere un ordinamento topologico è necessario che sia un **DAG** (vale a dire un grafo diretto aciclico).
Infatti la presenza di un ciclo nel grafo implica che nessuno dei nodi del ciclo possa comparire nell’ordine giusto; ognuno di essi richiede di apparire nell’ordinamento alla destra di chi lo precede.

In altre parole un grafo che non ha **sorgenti** non può avere un ordinamento topologico (sarebbe il primo nodo nell’ordinamento). Inoltre l’**ultimo nodo** dell’ordinamento deve essere necessariamente un **pozzo** (non può avere archi uscenti)
Grazie a questa proprietà è possibile costruire un ordinamento topologico dei nodi del DAG in questo modo:
- Inizio la sequenza dei nodi con una sorgente
- Cancello dal DAG quel nodo sorgente e le frecce che partono da lui (non creeranno problemi nel seguito), ottenendo un nuovo DAG
- Itero questo ragionamento finché non ho sistemato in ordine lineare tutti i nodi

```python
def sortTop(G):
	gradoEnt = [0]*len(G)
	for i in range(len(G)):
		for v in G[i]:
			gradoEnt[v]+=1
	sorgenti = [ i for i in range(len(G)) if gradoEnt[i] == 0 ]
	ST = []
	while sorgenti:
		u = sorgenti.pop()
		ST.append(u)
		for v in G[u]:
			gradoEnt[v]-=1
			if gradoEnt[v] == 0:
				sorgenti.append(v)
	if len(ST) == len(G):
		return ST
	return []
```
Complessità:
- inizializzare il vettore dei gradi entranti costa $O(n+m)$
- inizializzare l’insieme delle sorgenti costa $O(n)$
- il `while` viene iterato $O(n)$ volte e il costo totale del `for` al termine del `while` è $O(m)$
Il costo dell’algoritmo è $O(n+m)$

### Algoritmo alternativo
1. Effettua una visita DFS del DAG a partire dal nodo $0$
2. Man mano che termina la visita dei vari nodi, inseriscili in una lista
3. Restituisci come ordinamento dei nodi il `reverse` della lista

>[!info] Dimostrazione
>Siano $x$ e $y$ due nodi in $G$, con arco che va da $x$ a $y$. Consideriamo i due possibili casi e facciamo vedere che in entrambi i casi nella lista, prima di effettuare il reverse, $y$ prevede $x$
>- L’arco $(x,y)$ viene attraversato durante la visita → in questo caso banalmente la visita di $y$ finisce prima della visita di $x$ e $y$ finisce nella lista prima che ci finisca $x$
>- L’arco $(x,y)$ non viene attraversato durante la visita → durante la visita di $x$ il nodo $y$ è già visitato e la sua visita è anche già terminata (infatti da $y$ non c’è un cammino che porta a $x$, in caso contrario nel DAG ci sarebbe un ciclo), anche in questo caso $y$ finisce nella lista prima che ci finisca $x$

>[!example]
>![[Pasted image 20250311103303.png]]
>Ordine di fine visita → $4,3,2,1,5,0,6$
>Sort topologico → $6,0,5,1,2,3,4$

```python
def DFSr(u, G, visitati, lista):
	visitati[u] = 1
	for v in G[u]:
		if visitati[v] == 0:
			DFSr(v, G, visitati, lista)
	lista.append(u)

def sortTop1(G):
	visitati = [0]*len(G)
	lista = []
	for u in range(len(G)):
		if visitati[u] == 0:
			DFSr(u, G, visitati, lista)
	lista.reverse()
	return lista
```
Poiché si visitano sempre nodi diversi il costo della DFS è $O(n+m)$ (compreso il for), rimane solo da aggiungere il costo di `reverse`:
$$
O(n+m)+O(n)=O(n+m)
$$
