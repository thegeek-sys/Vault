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

L’idea di partenza **sbagliata** è: visita il grafo, e se nel corso della visita incontri un nodo già visitato interrompila e restituisci `True`, se al contrario la visita termina regolarmente restituisci `False`

Nei grafi non diretti l’algoritmo restituirebbe sempre `True` in quanto ogni arco nei grafi non diretti risulterebbe come due archi nelle direzioni opposte se fosse un grafo diretto (il nostro algoritmo troverebbe sempre un ciclo)
![[Pasted image 20250311104743.png|600]]

Per risolvere il problema, durante la visita alla ricerca del ciclo, devo distinguere nella lista di adiacenza di ciascun nodo $y$ che incontro il nodo $x$ che mi ha portato a visitarlo (il padre di $y$ nell’albero DFS)