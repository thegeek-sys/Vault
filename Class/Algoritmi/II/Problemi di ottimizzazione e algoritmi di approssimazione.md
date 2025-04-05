---
Created: 2025-04-05
Class: "[[Algoritmi]]"
Related:
---
---
## Problemi di ottimizzazione
Un problema di ottimizzazione è un tipo di problema in cui l’obbiettivo è trovare la migliore soluzione possibile tra un insieme di soluzioni ammissibili.
Ogni soluzione ammissibile, cioè una soluzione che soddisfa tutte le condizioni imposte dal problema, ha un valore associato che può essere un “costo” o un “beneficio”

A seconda del tipo di problema, l’obiettivo può essere minimizzare questo valore (es. ridurre i costi) o massimizzarlo (es. ottenere il massimo guadagno). 
Abbiamo quindi problemi di **minimizzazione** e problemi di **massimizzazione**

>[!example]
>Consideriamo il problema dello spanning tree (albero di copertura minimo) su un grafo pesato.
>Una soluzione ammissibile in questo caso è un albero di copertura, ovvero un sottoinsieme di archi che connette tutti i nodi del grafo senza formare cicli. La soluzione ottima è l’albero di copertura che ha il costo minimo, cioè la somma dei pesi degli archi è la più bassa possibile tra tutti gli alberi di copertura. In questo caso, trovare una soluzione ottima è un problema che può essere risolto in tempo polinomiale (es. Kruskal)
>
>Un altro esempio di problema di ottimizzazione è la ricerca del cammino minimo tra due nodi $a$ e $b$ in un grafo pesato.
>Una soluzione ammissibile è un cammino che collega $a$ e $b$ attraverso una sequenza di archi del grafo. La soluzione ottima, invece, è il cammino che ha il costo minimo, cioè la somma dei pesi degli archi del cammino è la più bassa possibile. Anche in questo caso trovare una soluzione ottima è un problema che può essere risolto in tempo polinomiale (es. Dijsktra)
>
>>[!warning]
>>E’ importante sottolineare che, sebbene nei due precedenti esempi la complessità  di trovare la soluzione ottima sia polinomiale, nella maggior parte dei problemi di ottimizzazione trovare una soluzione ottima risulta essere un compito molto più difficile.
>>In molti casi infatti, trovare una soluzione ottima può essere un problema in cui la complessità cresce esponenzialmente co la dimensione del problema
>>
>>In pratica, sebbene determinare se una soluzione è ammissibile possa essere fatto in tempo polinomiale, trovare la soluzione ottima richiede, nella maggior parte dei casi, algoritmi molto più complessi e intensivi in termini di risorse computazionali

---
## Algoritmi di approssimazione
Dato un grafo $G$ una sua **copertura tramite nodi** è un sottoinsieme $S$ dei suoi nodi tale che tutti gli archi di $G$ hanno almeno un estremo in $S$

![[Pasted image 20250405161824.png]]

>[!info] Il problema di ottimizzazione della copertura tramite nodi
>Dato un grafo non diretto $G$ trovare una copertura tramite nodi di minima cardinalità
>![[Pasted image 20250405161939.png|300]]

Un semplice approccio *greedy* al problema è il seguente: finché ci sono archi non coperti inserisci in $S$ il nodo che copre il **massimo** numero di archi ancora scoperti

Questa soluzione però non risulta essere corretta nel caso di un grafo di questo tipo
![[Pasted image 20250405162140.png]]
Infatti, tramite l’algoritmo prima descritto, al primo passo verrebbe inserito in $S$ il nodo $e$ (l’unico a coprire 4 archi) dopodiché tutti i nodi restanti sono in grado di coprire lo stesso numero di archi e nei passi successivi almeno un nodo per ogni coppia evidenziata nella foto seguente deve essere scelto
![[Pasted image 20250405162311.png|180]]

Ma la soluzione ottima usa solo 4 nodi:
![[Pasted image 20250405162341.png|180]]

