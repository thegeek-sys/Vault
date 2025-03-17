---
Created: 2025-03-17
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Problema
Abbiamo tre contenitori di capienza 4, 7 e 10 litri. Inizialmente i contenitori da 4 e 7 litri sono pieni d’acqua e quello da 10 è vuoto.
![[Pasted image 20250317115117.png|center|150]]

Possiamo effettuare un solo tipo di operazione: versare acqua da un contenitore ad un altro fermandoci quando il contenitore sorgente è vuoto o quello destinazione pieno.

>[!question] Problema
>Esiste una sequenza di operazioni di versamento che termina lasciando esattamente due litri d’acqua nel contenitore da 4 o nel contenitore da 7?

Posso modellare il problema con un grafo diretto $G$.
I nodi di $G$ sono i possibili stati di riempimento dei 3 contenitori; ogni nodo rappresenta una configurazione $(a,b,c)$ dove $a$ è il numero di litri nel contenitore da 4, $b$ il numero di libri nel contenitore da 7 e $c$ il numero di litri nel contenitore da 10.
Metto un arco dal nodo $(a,b,c)$ al nodo $(a',b',c')$ se dallo stato $(a,b,c)$ è possibile passare allo stato $(a',b',c')$ con un versamento lecito (dunque potrebbe non essere possibile passare al nodo $(a',b',c')$ al nodo $(a,b,c)$)

Ricordiamo però che possiamo escludere configurazioni a priori poiché per come è impostato il problema $a+b+c=11$

>[!example] Frammento del grafo $G$
>![[Pasted image 20250317115329.png|350]]

Per risolvere il nostro problema basterà chiedersi se nel grafo diretto $G$ almeno uno dei nodi $(2,?,?)$ o $(?,2,?)$ è raggiungibile a partire dal nodo $(4,7,0)$
