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
Possiamo quindi aggiungere al grafo un ulteriore nodo pozzo $(-1,-1,-1)$ con archi entranti che provengono dai nodi $(2,?,?)$ o $(?,2,?)$ e chiederci se il pozzo è raggiungibile da $(4,7,0)$
Per quanto riguarda il numero di nodi del grafo abbiamo $n=441$ infatti un contenitore con capienza $z$ può contenere $0, 1, 2,\dots, z$ litri d'acqua, esso può assumere $z + 1$ stati diversi. Quindi, il numero dei nodi è $5 \times 8 \times 11 = 440$ (questi comprendono anche nodi che dalla configurazione iniziale non possono essere raggiunti) più il nodo pozzo

Abbiamo quindi ridotto il nostro problema ad un problema di raggiungibilità di nodi su grafi, problema che siamo in grado di risolvere in tempo $O(n + m)$ con una visita DFS o una visita BFS.

Se siamo interessati a raggiungere una delle configurazioni target $(2,?, ?)$ o $(?,2,?)$ con il minor numero di travasi possiamo ancora risolvere il problema in $O(n + m)$ con una visita BFS per la ricerca dei cammini minimi calcolando le distanze minime a partire dal nodo $(4,7,0)$

>[!question] Consideriamo ora questa variante del problema
>Una sequenza di operazioni di versamento è buona se termina lasciando esattamente 2 litri o nel contenitore da 4 o nel contenitore da 7. Inoltre, diciamo che una sequenza buona è parsimoniosa se il totale dei litri versati in tutti i versamenti della sequenza è minimo rispetto a tutte le sequenze buone.
>**Voliamo trovare una sequenza che sia buona e parsimoniosa**

Dovendo misurare il numero di litri d’acqua versati nelle varie mosse, conviene assegnare un costo ad ogni arco: il **numero di litri che vengono versati** nella mossa corrispondente

>[!example] Frammento del grafo $G$
>![[Pasted image 20250317120203.png]]

Il problema diventa ora quello di trovare un cammino dal nodo $(4,7,0)$ al nodo pozzo $(-1,-1,-1)$ che minimizza la somma dei costi degli archi attraversati

>[!info]
>Questo problema è una generalizzazione del problema dei cammini di lunghezza minima perché gli archi invece di avere tutti lo stesso valore, cioè 1, hanno valori differenti

>[!hint]
>Possiamo sostituire un arco da $x$ a $y$ di costo $C$ con un cammino di $C-1$ nuovi nodi (chiamati nodi dummy)
>![[Pasted image 20250317120439.png]]
>
>Il problema di questo secondo modo però nella pratica non viene usato per due motivi:
>- spazio
>- non è possibile rappresentare pesi non interi (non scalabile)



