---
Created: 2025-04-30
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Link state
Lo stato di un link indica il costo associato al link. Se il costo è $\infty$ significa che il collegamento non esiste oppure è stato interrotto

Ogni nodo deve conoscere i costi di tutti i collegamenti della rete (nel distance vector invece si usavano soli i vicini). Il link state database mantiene la mappa completa della rete

---
## Link state database (LSDB)
Il link-state database è unico per tutta la rete e ogni nodo ne possiede una copia

![[Pasted image 20250430215936.png|300]]

Viene rappresentato tramite una matrice
![[Pasted image 20250430220006.png]]

### Come può un nodo costruire il LSDB?
Ogni nodo della rete deve innanzitutto conoscere i propri vicini e i costi dei collegamenti verso di loro. Per cui ogni nodo invia un messaggio di `hello` a tutti i suoi vicini di conseguenza ogni nodo riceve gli `hello` dei vicini e crea la lista dei vicini con i relativi costi dei collegamenti

La lista (vicino, costo) viene chiamata **LS packet** (*LSP*)
Si può dire dunque che ogni nodo esegue un *flooding* degli LSP:
- invia a tutti i vicini il proprio LSP
- quando riceve l’LSP di un vicino, se è un nuovo LSP allora lo inoltra a tutti i suoi vicini eccetto quello da cui lo ha ricevuto

>[!example]
>Gli LSP dei singoli nodi
>![[Pasted image 20250430223004.png|450]]
>
>Link-state database
>![[Pasted image 20250430223041.png|300]]

---
## Algoritmo d’instradamento a link state
In questo caso viene utilizzato l’**algoritmo di Dijkstra** che permette di calcolare il cammino minimo a costo minimo da un nodo a tutti gli altri nodi della rete (crea una tabella di inoltro per quel nodo)
Questo algoritmo è **iterativo**, dunque dopo la $k$-esima iterazione i cammini a costo minimo sono noti a $k$ nodi di destinazione. Viene quindi eseguito un numero di volte pari al numero di nodi nella rete

>[!info] Notazione
>- $N$ → insieme dei nodi della rete
>- $c(x,y)$ → costo del cammino minimo dal nodo origine alla destinazione $v$ per quanto riguarda l’iterazione corrente
>- $p(v)$ → immediato predecessore di $v$ lungo il cammino
>- $N'$ → sottoinsieme di nodi per cui il cammino a costo minimo dall’origine è definitivamente noto

### Pseudocodice
![[Pasted image 20250430224414.png]]

>[!example]- Esempio
>Dato il seguente grafo
>![[Pasted image 20250430224644.png|350]]
>
>![[Pasted image 20250430224601.png]]
>![[Pasted image 20250430224811.png]]
>![[Pasted image 20250430224825.png]]
>![[Pasted image 20250430224834.png]]
>![[Pasted image 20250430224922.png]]
>![[Pasted image 20250430224933.png]]
>![[Pasted image 20250430224947.png]]
>![[Pasted image 20250430225009.png]]
>![[Pasted image 20250430225022.png]]
>![[Pasted image 20250430225044.png]]
>
>Il grafo a costo minimo risulta quindi essere:
>![[Pasted image 20250430225116.png|350]]
>
>Tabella di inoltro in $u$ (per andare al nodo destinazione deve prima passare per la seconda posizione della tupla)
>![[Pasted image 20250430225305.png]]

---
## Confronto tra LS e DV
**Complessità dei messaggi**:
- LS → con $n$ nodi, $E$ collegamenti, implica l’invio di $O(nE)$ messaggi (ogni nodo deve conoscere il costo degli $E$ link)
- DV → richiede scambi tra nodi adiacenti (il tempo di convergenza può variare)

**Velocità di convergenza**:
- LS → l’algoritmo ha complessità $O(n^2)$ (procede prima $n$ nodi poi $n-1$, poi $n-2$ ecc.)
- DV → può convergere lentamente e può presentare cicli di instradamento e il problema del conteggio infinito

**Robustezza**: OSPF è più robusto di RIP

**Cosa avviene se un router funziona male?**
LS → un router può comunicare via broadcast un costo sbagliato per uno dei suoi collegamenti connessi (ma non per altri, i nodi si occupano soltanto di calcolare le poprie tabelle)
DV → un nodo può comunicare cammini a costo minimo errati a tutte le destinazioni e la tabella di ciascun nodo può essere usata da altri (un calcolo errato si può diffondere per l’intera rete)

---
## Protocollo OSPF
Il **protocollo OSPF** (*Open Shortest Path First*) è basato sull’algoritmo LS (utilizza il flooding e l’algoritmo di Dijsktra)

 >[!hint] Un protocollo è un qualcosa di più di un algoritmo
 >Un protocollo deve definire il suo ambito di funzionamento, i messaggi che vengono scambiati, la comunicazione tra router e l’interazione con gli altri protocolli
 
 Ogni volta che si verifica un cambiamento nello stato di un collegamento, il router manda informazioni d’instradamento a tutti gli altri router e inoltre invia periodicamente (ogni 30 minuti) messaggi OSPF all’intero sistema autonomo (utilizzando il flooding)

I messaggi OSPF vengono trasportati direttamente in datagrammi IP usando il numero di protocollo $89$ nel campo IP protocol