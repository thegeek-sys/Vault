---
Created: 2025-04-24
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Index
- [[#Recap Forwarding dei datagrammi IP forwarding datagrammi IP|Recap forwarding datagrammi IP]]
- [[#Introduction|Introduction]]
- [[#Grafo di una rete di calcolatori|Grafo di una rete di calcolatori]]
	- [[#Grafo di una rete di calcolatori#Costi|Costi]]
- [[#Algoritmo d’instradamento con vettore distanza|Algoritmo d’instradamento con vettore distanza]]
	- [[#Algoritmo d’instradamento con vettore distanza#Equazione di Bellman-Ford|Equazione di Bellman-Ford]]
		- [[#Equazione di Bellman-Ford#Rappresentazione grafica|Rappresentazione grafica]]
	- [[#Algoritmo d’instradamento con vettore distanza#Vettore distanza|Vettore distanza]]
- [[#Come viene creato il vettore delle distanze?|Come viene creato il vettore delle distanze?]]
- [[#Algoritmo con vettore distanza|Algoritmo con vettore distanza]]
	- [[#Algoritmo con vettore distanza#Guasto del collegamento|Guasto del collegamento]]
- [[#RIP|RIP]]
	- [[#RIP#Tabelle di routing|Tabelle di routing]]
	- [[#RIP#RIP protocol|RIP protocol]]
	- [[#RIP#Messaggi RIP|Messaggi RIP]]
		- [[#Messaggi RIP#Struttura|Struttura]]
	- [[#RIP#Timer RIP|Timer RIP]]
		- [[#Timer RIP#Guasto sul collegamento e recupero|Guasto sul collegamento e recupero]]
	- [[#RIP#Caratteristiche di RIP|Caratteristiche di RIP]]
	- [[#RIP#Implementazione di RIP|Implementazione di RIP]]
---
## Recap [[Forwarding dei datagrammi IP|forwarding datagrammi IP]]
Inoltrare significa collocare il datagramma sul giusto percorso (porta di uscita del router) che lo porterà a destinazione (o lo farà avanzare verso la prossima destinazione)

Quando un host ha un datagramma da inviare lo invia al router della rete locale. Quando un router riceve un datagramma da inoltrare accede alla tabella di routing per trovare il successivo hop a cui inviarlo.
L’inoltro richiede una riga della tabella per ogni blocco di rete

---
## Introduction

>[!question] Quale percorso deve seguire un pacchetto che viene instradato da un router sorgente a un router destinazione? Se sono disponibili più percorsi, quale si sceglie?

Il routing si occupa di trovare il percorso migliore e inserirlo nella tabella di routing

>[!warning] Il routing costruisce le tabelle, il forwarding le usa

---
## Grafo di una rete di calcolatori

>[!example]
>![[Pasted image 20250424230427.png|center|370]]
>$G=(N,E)$
>$N=\text{insieme dei nodi (router)}=\{u,v,w,x,y,z\}$
>$E=\text{insieme di archi}=\{(u,v), (u,x), (v,x), (v,w), (x,w), (x,y), (w,y), (w,z), (y,z)\}$
>
>Un path nel grafo $G=(N,E)$ è una sequenza di nodi $(x_{1},x_{2},\dots,x_{n})$ tale che ognuna delle coppie $(x_{1},x_{2}),(x_{2},x_{3}),\dots,(x_{n-1},x_{n})$ sono archi di $E$

### Costi
Nel grafo $c(x,x')$ è il costo del collegamento $(x,x')$ (es. $c(w,z)=5$), dunque il costo di un cammino è semplicemente la somma di tutti i costi degli archi lungo il cammino

>[!question] Cosa rappresenta il costo?
>- lunghezza fisica del collegamento
>- velocità del collegamento
>- costo monetario per poter attraversare il collegamento

Serve quindi un **algoritmo di instradamento** per poter determinare il cammino a costo minimo

---
## Algoritmo d’instradamento con vettore distanza
L’algoritmo d’instradamento presenta  due principali caratteristiche, è:
- **distribuito** → ogni nodo riceve informazione dai vicini e opera su quelle
- **asincrono** → non richiede che tutti i nodi operino al passo con gli altri

Si basa su:
1. equazione di Bellman-Ford
2. concetto di vettore di distanza

### Equazione di Bellman-Ford
Definisce $D_{x}(y):=\text{il costo del percoso a costo minimo dal nodo x al nodo y}$

Allora
$$
D_{x}(y)=\text{min}_{v}\{c(x,v)+D_{v}(y)\}
$$
dove $\text{min}_{v}$ riguarda tutti i vicini di $x$

#### Rappresentazione grafica
I percorsi $a\to b$, $b\to y$, $c\to y$ sono percorsi a costo minimo precedentemente stabiliti e $x\to y$ è un nuovo percorso a costo minimo

![[Pasted image 20250424234155.png|450]]
$$
D_{xy}=\text{min}\{(c_{xa}+D_{ay}),(c_{xb}+D_{by}),(c_{xc}+D_{cy})\}
$$

### Vettore distanza
Un albero a costo minimo è una combinazione di percorsi a costo minimo dalla radice dell’albero verso tutte le destinazioni

Il vettore di distanza è un array monodimensionale che rappresenta l’albero. Un vettore di distanza non fornisce il percorso da seguire per giungere alla destinazione ma solo i costi minimi per le destinazioni

---
## Come viene creato il vettore delle distanze?
Ogni nodo della rete quando viene inizializzato crea un vettore distanza iniziale con le informazioni che riesce ad ottenere dai propri vicini (nodi a cui è direttamente collegato)

Per creare il vettore dei vicini invia messaggi di `hello` attraverso le sue interfacce (e lo stesso fanno i vicini) e scopre l’identità dei vicini e la sua distanza da ognuno di essi
Dopo che ogni nodo ha creato il suo vettore ne invia una copia ai suoi vicini

>[!example]
>Vettori distanza iniziali dopo messaggi di hello
>![[Pasted image 20250424235444.png|400]]
>
>Quando un nodo riceve un vettore distanza un vicino provvede ad aggiornare il suo vettore distanza applicando l’equazione di Bellman-Ford
>
>Cosa succede quando $B$ riceve una copia di $A$?
>![[Pasted image 20250424235730.png|300]]
>
>Cosa succede se ora $B$ riceve una copia di $E$?
>![[Pasted image 20250424235818.png|300]]

---
## Algoritmo con vettore distanza
L’idea di base è che ogni nodo invia una copia del proprio vettore distanza a ciascuno dei suoi vicini. Quando un nodo $x$ riceve un nuovo vettore distanza $DV$ da qualcuno dei suoi vicini, lo salva e usa la formula Bellman-Ford per aggiornare il proprio vettore distanza come segue:
$$
D_{x}(y)\gets \text{min}_{v}(c(x,v)+D_{v}(y))
$$
per ciascun nodo $y$ in $N$

Se il vettore distanza del nodo $x$ è cambiato per via di tale passo di aggiornamento, il nodo $x$ manderà il proprio vettore distanza aggiornato a ciascuno dei suoi vicini, i quali a loro volta aggiornano il loro vettore distanza

![[Pasted image 20250425000618.png|250]]

### Guasto del collegamento
Sfruttando il fatto che ad ogni aggiornamento viene aggiornato il vettore delle distanze è possibile verificare la presenza di un guasto

>[!example] Si guarda il collegamento tra $A$ e $X$
>![[Pasted image 20250425002059.png]]
>al punto b. il router $A$ si aggiorna male a causa del guasto di $X$, quindi invia l’aggiornamento a $B$ che incrementa il vettore delle distanze. D’ora in poi la richiesta  di aggiornamento continuerà a rimbalzare tra $A$ e $B$ (di conseguenza aumenterà il costo) finché il costo non sarà $\infty$

Per poter evitare di entrare in questo loop infinito si hanno due possibilità:
- **split horizon** → se il nodo $B$ ritiene che il percorso ottimale per raggiungere il nodo $X$ passi attraverso $A$, allora non deve fornire questa informazione ad $A$ (l’informazione è arrivata da $A$ e quindi la conosce già). Come risultato si ha che $B$ elimina la riga di $X$ dalla tabella prima di inviarla ad $A$
- **poisoned reverse** → se un nodo usa un vicino per raggiungere una destinazione (ormai guasta), segnala attivamente a quel vicino che il percorso ha costo $\infty$, così il vicino non sarà tentato di usarlo per quella destinazione.

---
## RIP
Il **RIP** (*Routing Information Protocol*) è un protocollo a vettore di distanza ed è tipicamente incluso in UNIX BSD dal 1982
La distanza viene misurata in hop ed ha un massimo di $15$ hop (il valore $16$ indica l’infinito)

![[Pasted image 20250425003656.png]]

### Tabelle di routing
L’informazione della tabella di routing è sufficiente per raggiungere la destinazione
![[Pasted image 20250425003842.png]]

### RIP protocol
Periodicamente, ogni router che esegue RIP invia le righe della propria tabella di routing (vettore di distanza) per fornire informazioni agli altri router sulle reti e sugli host che è in grado di raggiungere
Qualsiasi router sulla stessa rete di quello che invia queste informazioni potrà aggiornare la propria tabella in base alle informazioni ricevute
Un router che riceve un messaggio da un altro router sulla stessa rete, in cui si dice che può raggiungere la rete $X$ con un costo $N$, sa di poter raggiungere la rete $X$ con un costo pari a $N+1$ inviando i pacchetti al router da cui ha ricevuto il messaggio

>[!warning]
>Invece di inviare solo i vettori di distanza, i router inviano l’intero contenuto della tabella di routing

### Messaggi RIP
RIP si basa su una coppia di processi client-server e sul loro scambio di messaggi (protocollo di livello rete, implementato a livello applicazione)

Si hanno due tipi di messaggi RIP:
- **RIP Request** → quando un nuovo router viene inserito nella rete invia una RIP Request per ricevere immediatamente informazioni di routing
- **RIP Response** (o *advertisements*) → si ha o in risposta una Request (solicited response) oppure periodicamente ogni $30$ secondi (unsolicited response)

Ogni messaggio contiene un elenco comprendente fino a $25$ sottoreti di destinazione all’interno del sistema autonomo, nonché la distanza del mittente rispetto a ciascuna di tali sottoreti

#### Struttura
![[Pasted image 20250425004840.png]]

### Timer RIP
Il protocollo RIP presenta diversi timer:
- timer periodico → controlla invio di messaggi di aggiornamento (25-35 secondi)
- timer di scadenza → regola la validità dei percorsi (180 secondi); se entro lo scadere del timer non si riceve aggiornamento, il percorso viene considerato scaduto e il suo costo impostato a $16$
- timer per garbage collection → elimina percorsi dalla tabella (120 secondi); quando le informazioni non sono più valide, il router continua ad annunciare il percorso con costo pari a 16, e, allo scadere del timer, rimuove il percorso

#### Guasto sul collegamento e recupero
Se un router non riceve notizie dal suo vicino per 180 sec allora il nodo adiacente/il collegamento viene considerato spento o guasto. Quindi:
- RIP modifica la tabella d’instradamento locale
- RIP propagare l’informazione mandando annunci ai router vicini
- i vicini inviano nuovi messaggi (se la loro tabella d’instradamento è cambiata)
- l’informazione che il collegamento è fallito si propaga rapidamente su tutta la rete.
- l’utilizzo del poisoned reverse evita i loop ($\text{distanza infinita} = 16 \text{ hop}$)

### Caratteristiche di RIP
- **split horizon with poisoned reverse** (inversione avvelenata)
	- serve per evitare che un router invii rotte non valide al router da cui ha imparato la rotta (evitare cicli)
	- si imposta a infinito ($16$) il costo della rotta che passa attraverso il vicino a cui si manda l’advertisement
- **triggered updates**
	- riduce il problema della convergenza lenta
	- quando cambia una rotta si inviano immediatamente informazioni ai vicini senza attendere il timeout.
- **hold-down**
	- fornisce robustezza
	- quando si riceve una informazione di una rotta non più valida, si avvia un timer e tutti gli advertisement riguardanti quella rotta che arrivano entro il timeout vengono tralasciati

### Implementazione di RIP
Il protocollo RIP è implementato nel livello applicazione tramite UDP (porta 520)
Un processo chiamato *routed* (*route daemon*) esegue RIP, ossia mantiene le informazioni di instradamento e scambia messaggi con i processi routed nei router vicini

Poiché RIP viene implementato come un processo a livello applicazione, può inviare e riceve messaggi su una socket standard e utilizzare un protocollo di trasporto standard

![[Pasted image 20250425010334.png]]