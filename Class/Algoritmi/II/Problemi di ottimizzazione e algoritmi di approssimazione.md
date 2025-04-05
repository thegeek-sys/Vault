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

Ci sono moltissimi problemi di ottimizzazione come copertura tramite nodi che sono computazionalmente difficili. Infatti per questi problemi non si conoscono algoritmi neanche lontanamente efficienti (sostanzialmente per questi problemi sono noti solo algoritmi esponenziali)

### Euristiche
In questi casi potrebbe essere già soddisfacente ottenere una soluzione **ammissibile** che sia soltanto “vicina” ad una soluzione ottima e, ovviamente, più è vicina meglio è
Fra gli algoritmi che non trovano sempre una soluzione ammissibile ottima, è importante distinguere due categorie piuttosto differenti:
- **algoritmi di approssimazione**
- **euristiche**

Gli **algoritmi di approssimazione** sono algoritmi per cui si dimostra che la soluzione ammissibile prodotta ha una certa “vicinanza” alla soluzione ottima. In altre parole, è garantito che la soluzione prodotta approssima entro un certo grado la soluzione ottima

Le **euristiche** sono algoritmi per cui non si riesce a dimostrare che la soluzione ammissibile prodotta ha sempre una certa vicinanza ad una soluzione ottima. Però sperimentalmente sembrano comportarsi bene. Sono l’ultima spiaggia quando non si riesce a trovare algoritmi corretti efficienti né algoritmi di approssimazione efficienti che garantiscono un buon gradi di approssimazione

Per una gran parte dei problemi computazionalmente difficili non solo non si conoscono algoritmi corretti efficienti ma neanche buoni algoritmi di approssimazione. Non è quindi sorprendente che fra tutti i tipi di algoritmi, gli algoritmi euristici costituiscono la classe più ampia e che ha dato luogo ad una lettura sterminata

---
## Minimizzazione
Un algoritmo di approssimazione per un dato problema è un algoritmo per cui si dimostra che la soluzione prodotta a**pprossima sempre entro un certo grado una soluzione ottima** per il problema. Si tratta quindi di specificare cosa si intende per "approssimazione entro un certo grado".

Iniziamo con problemi di **minimizzazione** dove ad ogni soluzione ammissibile è associato un costo e cerchiamo quindi la soluzione ammissibile di costo minimo

Il modo usuale di misurare il gradi di approssimazione è il rapporto tra il costo della soluzione prodotta dall’algoritmo e il costo della soluzione ottima

Più formalmente si dice che $A$ approssima il problema di minimizzazione entro un fattore di approssimazione $\rho$ se *per ogni istanza* $I$ del problema vale:
$$
\frac{A(I)}{OPT(I)}\leq \rho
$$
Dove con $OPT(I)$ si indica il costo di una soluzione ottima per l’istanza $I$ e con $A(I)$ il costo della soluzione prodotta dall’algoritmo $A$ per quell’istanza

>[!hint]
>Per problemi di massimizzazione dove ad ogni soluzione ammissibile è associato un valore si considera il rapporto inverso vale a dire:
>$$\frac{OPT(I)}{A(I)}$$

>[!info]
>Nota che, trattandosi di un problema di minimizzazione, risulta sempre $A(I)\geq OPT(I)$, di conseguenza il rapporto di approssimazione $\rho$ è sempre un numero maggiore o uguale a $1$
>- se $A$ approssima $P$ con fattore $1$, allora $A$ è corretto per $P$ perché trova sempre una soluzione ottima
>- se $A$ approssima $P$ entro un fattore $2$, allora $A$ trova sempre una soluzione di costo al più doppio di quello della soluzione ottima
>
>Ovviamente quanto più il rapporto di approssimazione è vicino ad $1$ tanto più l’algoritmo d’approssimazione è buono

---
## Rapporto d’approssimazione - Ricoprimento tramite nodi
Proviamo a valutare il rapporto d’approssimazione dell’algoritmo greedy visto prima per il problema del ricoprimento tramite nodi

Abbiamo notato che l’algoritmo greedy non è corretto, infatti abbiamo trovato un’istanza per cui l’algoritmo produce una soluzione con $5$ nodi mentre la soluzione ottima ha $4$ nodi. Da ciò possiamo dedurre che il fattore di approssimazione è almeno $\frac{5}{4}>1$, ma **potrebbe essere peggiore**

Preso infatti questo grafo
![[Pasted image 20250405223345.png]]
Il rapporto di approssimazione diventa di $\frac{4}{3}$
![[Pasted image 20250405223425.png|500]]
In effetti per ogni costante $R$ si possono esibire grafi per cui l’algoritmo sbaglia di un fattore superiore a $R$. Quindi l’algoritmo greedy in esame non garantisce nessun fattore di approssimazione costante (sarà $\Omega(\log l)$)

### Soluzione migliorata
Ovviamente il fatto d’aver dimostrato che per un problema un certo algoritmo d’approssimazione ha un cattivo rapporto d’approssimazione non impedisce che per il problema possano esistere altri algoritmi d’approssimazione con un fattore d’approssimazione costante.

Consideriamo il seguente algoritmo greedy per la copertura di nodi: considera i vari archi del grafo uno dopo l’altro e ogni volta che ne trovi uno non coperto (vale a dire nessuno dei suoi estremi è in $S$) aggiungi entrambi gli estremi dell’arco alla soluzione $S$

Questa soluzione sicuramente produce una copertura, ma non è detto che sia minima (nel grafo con due soli nodi e un arco il rapporto d’approssimazione è $2$)

>[!info] Dimostrazione
>Dimostreremo qui che il rapporto d’approssimazione dell’algoritmo greedy è limitato a $2$
>