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

