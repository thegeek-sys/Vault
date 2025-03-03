---
Created: 2025-03-03
Class: "[[Algoritmi]]"
Related: 
Completed:
---
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
Un grafo **connesso** (esiste almeno un cammino tra ogni coppia di nodi) e senza **cicli** è detto **albero**


>[!example] Grafo sconnesso con ciclo
>![[Pasted image 20250303111808.png|200]]

