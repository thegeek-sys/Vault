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
1. 