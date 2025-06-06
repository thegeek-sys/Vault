---
Created: 2025-03-17
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
!---
## Index
- [[#Problema|Problema]]
- [[#Algoritmo di Dijkstra|Algoritmo di Dijkstra]]
	- [[#Algoritmo di Dijkstra#Implementazione tramite lista|Implementazione tramite lista]]
	- [[#Algoritmo di Dijkstra#Implementazione tramite heap|Implementazione tramite heap]]
	- [[#Algoritmo di Dijkstra#Le due implementazioni a confronto|Le due implementazioni a confronto]]
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
>
>In questa rappresentazione tutti gli archi valgono 1. E’ come se ogni arco corrisponda al versamento di un litro d’acqua. Così contando semplicemente gli archi di un cammino tra due nodi configurazione contiamo proprio il numero totale di litri versati nelle mosse relative al cammino. Quindi per risolvere il problema possiamo eseguire una BFS nel nuovo grafo a partire dal nodo $(4,7,0)$ che si ferma non appena trova il nodo $(1,1,1)$
>L’algoritmo avrà complessità $O(n'+m')$ dove $n'$ ed $m'$ sono i nodi e gli archi del nuovo grafo rispettivamente
>Nel nostro caso il peso degli archi non poteva superare 7, si aveva quindi $n'<7n$ ed $m'\leq 7m$ dove $n$ ed $m$ sono i nodi e gli archi del grafo pesato, rispettivamente
>
>Abbiamo quindi ricondotto un problema di cammini minimi su grafi pesati a quello dei cammini minimi in un grafo non pesato

---
## Algoritmo di Dijkstra

>[!question] Problema
>Dato un grafo pesato vogliamo trovare i cammini minimi e quindi anche le distanze da un certo nodo  $s$ (detto sorgente) a tutti gli altri nodi

![[Pasted image 20250319184703.png]]
I cammini minimi (in grassetti) e le distanze (in rosso) quando la sorgente è il nodo $0$

Per applicare questo algoritmo bisogna:
- costruire l’albero dei cammini minimi un arco per volta partendo dal nodo sorgente nel modo seguente
	- ad ogni passo aggiungi all’albero l’arco che produce il nuovo cammino più economico
	- alla nuova destinazione assegna come distanza il costo del cammino

>[!example]- Esempio applicazione algoritmo
>Nelle seguenti immagini in verde saranno i possibili archi da attraversare. I costi sono calcolati con il costo del nodo in cui mi trovo + costo dell’arco da attraversare
>
>![[Pasted image 20250319185650.png|300]]
>Al primo passo è possibile scegliere se andare in 5 (costo 4) o in 1 (costo 17). Scelgo il nodo 5 (costo 4)
>![[Pasted image 20250319185800.png|300]]
>Ora si può andare in 3 (costo 5), in 1 (costo 10) o in 1 (costo 17). Scelgo di andare in 3 (costo 5) tramite il nodo 5
>![[Pasted image 20250319190022.png|300]]
>Ora si può andare in 1 (costo 17), in 1 (costo 10), in 4 (costo 9) o in 2 (costo 17). Scelgo di andare in 4 (costo 9) tramite il nodo 3
>![[Pasted image 20250319190123.png|300]]
>Ora si può andare in 1 (costo 17), in 1 (costo 10), in 1 (costo 14), in 2 (costo 19) o in 2 (costo 17). Scelgo di andare in 1 (costo 10) tramite il nodo 5
>![[Pasted image 20250319190321.png|300]]
>Ora si può andare in 2 (costo 17) o in 2 (costo 19). Scelgo di andare in 2 (costo 17) tramite il nodo 3
>![[Pasted image 20250319190408.png|300]]
>
>Sono quindi terminati i nodi da poter esplorare, termina quindi l’algoritmo. L’albero costruito da questa visita sarebbe del tipo:
>![[Pasted image 20250319190758.png|200]]

Per come è strutturato questo algoritmo rientra perfettamente nel **paradigma della tecnica greedy**. Si tratta infatti di una sequenza di **decisioni irrevocabili**: ad ogni passo viene deciso il cammino (e quindi la distanza) dal nodo sorgente ad un nuovo nodo e questa decisione non viene più modificata

Le decisioni vengono prese in base ad un **criterio “locale”**: tra tutti i nuovi cammini che puoi trovare estendendo i vecchi di un arco prendi quello che costa meno. Ma questa tecnica spesso non porta a decisioni corrette a livello globale ma ci permette di ottenere soluzioni sub-ottimali

Lo pseudo-codice dell’algoritmo di Dijkstra:
$$
\begin{flalign}
&\text{Dijkstra}(s,G): \\
&P[0\dots n-1] \text{ vettore dei padri inizializzato a -1}\\
&D[0\dots n-1] \text{ vettore delle distanze inizializzato a }+\infty\\ \\
&D[s], P[s]=0,s\\ \\
&while \text{ esistono archi } \{x,y\} \text{ con } P[x]\neq-1 \text{ e }P[y]==-1: \\
&\qquad \text{sia }\{x,y\} \text{ quello per cui è minimo } D[x]+peso(x,y)\\
&\qquad D[y], P[y] = D[x] + peso(x,y), x\\
&return\; P,D
\end{flalign}
$$

La prima cosa da notare è che l’algoritmo **non è corretto nel caso di grafi con pesi anche negativi**
![[Pasted image 20250319192054.png]]
Al centro la soluzione prodotta da Dijkstra per il grafo $G$, a destra la soluzione corretta

>[!done] Dimostrazione correttezza nel caso di pesi positivi
>Ad ogni iterazione del $while$ viene assegnata una nuova distanza ad un nodo.
>Per induzione sul numero di iterazione mostreremo che la distanza assegnata è quella minima
>
>Il caso base è banale (al passo $0$ viene assegnata distanza zero alla sorgente e con pesi positivi non può esserci una distanza inferiore).
>
>Sia $T_{i}$ l’albero dei cammini minimi costruito fino al passo $i>0$ e $(u,v)$ l’arco aggiunto al passo $i+1$. Faremo vedere che $D[v]$ è la distanza minima di $v$ da $s$. Baserà mostrare che il costo di un eventuale cammino alternativo è sempre superiore o uguale a $D[v]$
>
>![[Pasted image 20250319215140.png]]
>
>Sia $C$ un qualsiasi cammino da $s$ a $v$ alternativo a quello presente nell’albero e $(x,y)$ il primo arco che incontriamo percorrendo il cammino $C$ all’indietro tale che $x$ è nell’albero $T_{i}$ e $y$ no (tale arco deve esistere perché $s$ è in $T_{i}$ mentre $v$ no)
>
>Per ipotesi induttiva $\text{costo}(C)\geq D[x]+peso(x,y)$
>>[!info]
>>Quest’affermazione è vera perché i pesi del grafo sono tutti non negativi
>
>L’algoritmo ha preferito estendere l’albero $T_{i}$ con l’arco $(u,v)$ anziché l’arco $(x,y)$ e in base alla regola con cui l’algoritmo sceglie l’arco con cui estendere l’albero deve quindi aversi $D[x]+peso(x,y)\geq D[u]+peso(u,v)$
>Da cui segue: $\text{costo}(C)\geq D[x]+peso(x,y) \geq D[u]+peso(u,v)=D[v]$
>
>Il cammino alternativo ha un costo superiore a $D[v]$

In un grafo pesato ogni arco ha associato un peso. Per rappresentare questo tipo di grafi per l’arco $(x,y)$ di peso $c$ nella lista di adiacenza di $x$ invece che il solo nodo di destinazione $y$ ci sarà la coppia $(y,c)$ con l’informazione sul nodo destinazione e il peso dell’arco

>[!example]
>![[Pasted image 20250319215505.png|400]]
>
>Ad esempio il grafo pesato $G$ in figura viene codificato come segue:
>$$\begin{align}G=[& \\&[(1,17),(5,4)], \\&[(0,17),(4,5),(5,6)], \\&[(3,12),(4,10)], \\&[(2,12),(4,4),(5,1)], \\&[(1,5),(2,10),(3,4)], \\&[(0,4),(1,6),(3,1)] \\]\end{align}$$

### Implementazione tramite lista
Nel vettore $\text{Lista}$, per ogni nodo $x$ memorizziamo una terna nella forma $\text{(definitivo, costo, origine)}$. Ecco cosa rappresenta ciascun elemento della terna per il nodo $x$:
- **definitivo**
	- è un flag che assume il valore $1$ se il costo per raggiungere $x$ è stato “definitivamente” stabilito, ossia se l’algoritmo ha confermato che non è possibile ottenere un percorso migliore a parte della sorgente
	- se vale $0$, significa che il costo per $x$ è ancora in fase di aggiornamento (non definitivo)
- **costo**
	- rappresenta il costo corrente minimo noto per raggiungere $x$ dalla sorgente $s$
	- all’inizio, per ogni nodo diverso da $s$ se questo valore è inizializzato a $\infty$, e per $s$ a $0$. Durante l’esecuzione dell’algoritmo, questo valore può essere aggiornato quando si trova in un percorso migliore
- **origine**
	- indica il nodo “padre” o “predecessore” lungo il cammino minimo dalla sorgente $s$ a $x$
	- se non è ancora stato trovato un percorso per $x$ oppure $x$ non ha ancora un predecessore, questo valore è inizialmente impostato a $-1$

In sintesi, la terna $\text{(definitivo, costo, origine)}$ associata al nodo $x$ in $\text{Lista}$ contiene tutte le informazioni necessarie per sapere se il cammino minimo verso $x$ è stato determinato, qual è il costo di tale cammino, e da quale nodo si giunge a $x$ lungo il percorso minimo

All’inizio l0unico nodo nell’albero è la sorgente, di conseguenza la lista è inizializzata come segue:
$$
\text{Lista}[x]=
\begin{cases}
(1,0,s)&\text{se } x=s \\
(0, costo, s)&\text{se }(costo,x)\in G[s] \\
(0,+\infty,-1)&\text{altrimenti}
\end{cases}
$$

Seguono una serie di iterazioni dove vengono eseguiti i seguenti passaggi:
1. **Selezione del nodo con costo minimo non definitivo**: si scorre l’intera struttura $\text{Lista}$ per individuare il nodo $x$ che non è ancora definitivo (cioè, il cui flag è $0$) e che ha il costo corrente minimo. Questo nodo rappresenta il candidato per il quale il cammino minimo dalla sorgente è attualmente noto
2. **Verifica di terminazione**: se il costo trovato è $\infty$, significa che non esistono altri nodi raggiungibili non ancora definitivi. In tal caso, il ciclo si interrompe
3. **Marcare il nodo $x$ come definitivo**: il nodo $x$ selezionato viene aggiornato in $\text{Lista}$ impostando il flag a $1$, indicando che il suo costo definitivo è stato fissato e non verrà più modificato
4. **Aggiornamento dei vicini di $x$**: per ogni nodo $y$ adiacente a $x$, se $y$ non è ancora definitivo e il nuovo costo ottenuto passato per $x$ (cioè $\text{costo}(x)+\text{costo}(x,y)$) è inferiore al costo attuale memorizzato per $y$, allora si aggiorna la terna corrispondente a $y$ in $\text{Lista}$: $$\text{Lista}[y]=(0,\text{costo}(x)+\text{costo}(x,y),x)$$
```python
def dijkstra(s, G):
	n = len(G)
	Lista = [(0, float('inf'), -1)]*n
	Lista[s] = (1, 0, s)
	
	for y, costo in G[s]:
		# aggiorno vicini di s
		Lista[y] = (0, costo, s)
	
	while True:
		minimo, x = float('inf'), -1
		# nodo non definitivo con costo minore
		for i in range(n):
			if Lista[i][0] == 0 and Lista[i][1] < minimo:
				minimo, x = Lista[i][1], i
		
		# non ci sono più nodi raggiunbili non definitivi
		if minimo == float('inf'):
			break
		
		# rendi definitivo il nodo x
		definitivo, costo_x, origine = Lista[x]
		Lista[x] = (1, costo_x, origine)
		
		# aggiornamento vicini
		for y, corso_arco in G[x]:
			if Lista[y][0] == 0 and minimo+costo_arco < Lista[y][1]:
				# y non definitivo e passando per x c'è un cammino
				# migliore
				Lista[y] = (0, minimo+costo_arco, x)
	
	# estrae i vettori delle distanze e dei padri
	D,P = [costo for _,costo,_ in Lista], [origine for _,_,origine in Lista]
	return D, P
```
Il costo delle istruzioni al di fuori del $while$ è $\Theta (n)$. Il $while$ viene eseguito al più $n-1$ volte (ad ogni iterazione un nuovo nodo viene selezionato e reso definitivo). All’interno del $while$ c’è:
- un primo $for$ che viene iterato esattamente $n$ volte
- un secondo $for$ che viene eseguito al più $n$ volte (tante volte quanti sono gli adiacenti presenti nella lista del nodo appena inserito nell’albero)

Il costo del $while$ è dunque $\Theta(n^2)$ e questa è anche la complessità dell’implementazione

>[!hint]
>Questa implementazione è ottima nel caso di grafi densi dove $m = \Theta(n^2)$

### Implementazione tramite heap
Sostituendo il vettore lista con un **heap minimo** potremmo estrarre l’elemento minimo in tempo logaritmico nel numero di elementi presenti nell’heap

L’idea è di mantenere un heap minimo contenente triple $(costo, u, v)$ dove $u$ è un nodo già inserito nell’albero dei cammini minimi e $costo$ rappresenta la distanza che si avrebbe qualora il nodo $y$ venisse inserito nell’albero dei cammini minimi attraverso il nodo $x$.

In questo modo, ad ogni estrazione dall’heap possiamo individuare in tempo logaritmico nella dimensione dell’heap il nodo $v$ da inserire nell’albero, il costo da assegnargli come distanza da $s$ e il padre $u$ a cui collegarlo.

Ogni volta che aggiungiamo un nodo $x$ all’albero, aggiorniamo anche l’heap inserendo, per ogni vicino $y$ di $x$, una nuova tripla $(DistanzaAggiornata, x, y)$. Poichè ogni inserimento ha costo logaritmico nella dimensione dell’heap, il tempo complessivo rimane gestibile.

Dato che non rimuoviamo elementi già presenti nell’heap, possono esistere più entry dello stesso nodo $y$ con distanze differenti. Tuttavia, la prima volta che il nodo $y$ viene estratto dall’heap, questa corrisponde necessariamente alla distanza minima calcolata fino a quel punto e le estrazioni successive di $y$ possono essere trascurate. Quindi ad ogni estrazione di un nodo, controlliamo come prima cosa se esso è giò stato aggiunto all’albero; in tal caso, l’informazione estratta viene ignorata.

```python
from heapq import heappush, heappop

def dijkstra1(s, G):
	n = len(G)
	D = [float('inf')]*n
	P = [-1]*n
	D[s] = 0
	P[s] = s
	H = [] # min-heap
	
	# inizializzazione heap con vicini di s
	for y, costo in G[s]:
		heappush(H, (costo, s, y))
	
	while H:
		# estrai nodo con distanza minore
		costo, x, y = heappop(H)
		if P[y] == -1:
			P[y] = x
			D[y] = costo
			
			for v, peso in G[y]:
				if P[v] == -1:
					heappush(H, (D[y]+peso, y, v))
	
	return D, P
```
Prima dell’accesso al while abbiamo l’inizializzazione di $D$ e $P$ per un costo $\Theta(n)$ e l’inserimento dei vicini di $s$ nell’heap per un costo di $O(n \log n)$. Abbiamo poi un $while$ con all’interno un $for$.

Ad ogni iterazione del $while$ si elimina un elemento da $H$ e eventualmente per mezzo del $for$ annidato si scorre la lista di adiacenza di un nodo e vengono aggiunti elementi ad $H$. Ogni lista di adiacenza può essere scorsa al più una volta quindi ad $H$ in totale possono essere aggiunti al più $O(m)$ elementi. Per quanto detto il numero di iterazioni del $while$ è $O(m)$.

Senza tener conto del $for$ annidato (i cui costi verranno calcolati a parte al prossimo punto), il costo di ciascuna iterazione del $while$ richiede $O(\log n)$ a causa dell’estrazione da $H$. Quindi il costo totale del while, senza tener conto del $for$, sarà $O(m \log m)$

Il tempo totale richiesto dalle varie iterazioni del $for$ annidato nel $while$ è $O(m \log n)$ in quando ad ogni iterazione scorro una lista di adiacenza diversa. In totale scorrerò $O(m)$ archi e per ogni arco pagherò $O(\log n)$ per inserimento in $H$.

La complessità di questa implementazione è dunque:
$$
O(n\log n)+O(m\log n)+O(m\log n)=O((n+m)\log n)
$$

### Le due implementazioni a confronto
Abbiamo visto due implementazioni dell’algoritmo di Dijkstra: la prima richiede $O(n^2)$ tempo, la seconda $O((n + m) \log n)$ .
La **prima** (rappresentazione tramite liste) è ottimale nel caso di **grafi densi**.
La **seconda** (rappresentazione tramite heap) è da preferirsi nel caso di **grafi sparsi**, presentando in quel caso una complessità $O(n \log n)$ mentre andrebbe evitata nel caso di grafi densi dove la complessità risultante sarebbe $O(n^2 \log n)$

Esistono implementazioni più e cienti di quest’algoritmo ottenute utilizzando strutture dati più sofisticate. Ad esempio utilizzando gli heap di Fibonacci il tempo d’esecuzione scende a $O(m + n \log n)$.
