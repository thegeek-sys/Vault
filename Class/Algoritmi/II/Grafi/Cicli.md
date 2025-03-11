---
Created: 2025-03-11
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction
Dato un grafo $G$ (diretto o indiretto) ed un suo nodo $u$ vogliamo sapere se da $u$ è possibile raggiungere un ciclo in $G$

![[Pasted image 20250311104046.png]]

---
## Errori
L’idea di partenza **sbagliata** è: visita il grafo, e se nel corso della visita incontri un nodo già visitato interrompila e restituisci `True`, se al contrario la visita termina regolarmente restituisci `False`

Nei grafi non diretti l’algoritmo restituirebbe sempre `True` in quanto ogni arco nei grafi non diretti risulterebbe come due archi nelle direzioni opposte se fosse un grafo diretto (il nostro algoritmo troverebbe sempre un ciclo)
![[Pasted image 20250311104743.png|600]]

Per risolvere il problema, durante la visita alla ricerca del ciclo, devo distinguere nella lista di adiacenza di ciascun nodo $y$ che incontro, il nodo $x$ che mi ha portato a visitarlo (non devo continuare la ricerca su $y$ se il prossimo nodo $x$ è il padre)

Ma anche in questo caso l’algoritmo risulterebbe essere **scorretto** nel caso di grafi diretti. Infatti incontrare in un grafo diretto un nodo già visitato non significa necessariamente che si è in presenza di un ciclo (la procedure può terminare con `True` anche se in assenza di ciclo)
![[Pasted image 20250311105608.png]]

---
## Algoritmo
Durante la visita DFS posso incontrare nodi già visitati in tre modi diversi:
- **archi in avanti** → frecce dirette da un antenato ad un discendente
- **archi all’indietro** → frecce dirette da un discendente ad un altenato
- **archi di attraversamento**

>[!example]
>![[Pasted image 20250311105859.png|600]]

**Solo** la presenza di **archi all’indietro** testimonia la presenza di un **ciclo**

Per risolvere il problema, durante la visita DFS alla ricerca del ciclo, devo poter distinguere la scoperta di nodi già visitati grazie ad un arco all’indietro dagli altri.
Posso individuare i visitati all’indietro notando che **solo nel caso di archi all’indietro la visita del nodo ha già terminato la sua ricorsione**

Per il vettore $V$ dei visitati uso tre step:
- in $V$ un nodo vale $0$ se il nodo non è stato ancora visitato
- in $V$ un nodo vale $1$ se il nodo è stato visitato ma la ricorsione su quel nodo non è ancora finita
- in $V$ un nodo vale $2$ se il nodo è stato visitato e la ricorsione su quel nodo è finita
In questo modo scopro un ciclo quando trovo un arco diretto verso un nodo già visitato che si trova nello stato $1$

```python
def DFSr(u, G, visitati):
	visitati[u] = 1
	for v in G[u]:
		if vistati[v] == 1:
			# ciclo trovato (nodo già visitato)
			return True
		if visitati[v] == 0:
			# non visitato, continua DFS
			if DFSr(v, G, visitati):
				return True
	visitati[u] = 2 # nodo completamete esplorato
	return False

def cicloD(u, G):
	visitati = [0]*len(G)
	return DFSr(u, G, visitati)
```
La complessità di questo algoritmo sarà: $O(n+m)$

Se voglio sapere se un grafo contiene un ciclo o meno, devo visitarlo tutto, non importa il punto da cui parto. Non è quindi difficile modificare la procedure appena viste senza alterarne la complessità $O(n+m)$

Di seguito la procedura modificata nel caso di grafi diretti:
```python
def cicloD(G):
	visitati = [0]*len(G)
	for u in range(len(G)):
		if visitati[u] == 0:
			if DFSr(u, G, visitati):
				return True
	return False
```
