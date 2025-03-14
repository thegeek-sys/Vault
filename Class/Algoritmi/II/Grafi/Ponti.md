---
Created: 2025-03-14
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction
In un grafo la connessione è una proprietà che può andar persa con la perdita di un arc

![[Pasted image 20250314102426.png|center|350]]

>[!info] Definizione
>Una arco la cui eliminazione disconnette il grafo è detto **ponte**

I ponti rappresentano criticità del grafo ed è quindi utile identificarli

---
## Determinare l’insieme dei ponti del grafo
Iniziamo ricordando che il grafo può anche non avere nessun ponte (cicli) così come avere tutti i suoi archi come ponti (in un albero qualunque arco è un ponte)

![[Pasted image 20250314102655.png|center|500]]

Una prima soluzione è basata sulla ricerca esaustiva, ovvero provare per ogni arco del grafo se questo è ponte o meno.
Per verificare se un arco $(a,b)$ è un ponte per $G$ richiede $O(m)$. Infatti basterebbe eliminare l’arco $(a,b)$ da $G$ e, con una visita DFS controllare se $b$ è raggiungibile da $a$. In totale un algoritmo del genere ha complessità $m\cdot O(m)=O(m^2)=O(n^4)$

Vedremo ora che questo problema è risolvibile in $O(m)$; l’idea è quella di usare un’unica visita DFS opportunamente modificata

I ponti vanno **ricercati unicamente tra gli $n-1$ archi dell’albero DFS**. Infatti un arco non presente nell’albero DFS non può essere ponte (se lo elimino gli archi dell’albero DFS continuano )