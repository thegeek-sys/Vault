---
Created: 2024-12-18
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Schedule legale|Schedule legale]]
- [[#Lock binario|Lock binario]]
	- [[#Lock binario#Esempio|Esempio]]
- [[#Modello per le transazioni|Modello per le transazioni]]
- [[#Equivalenza|Equivalenza]]
- [[#Schedule serializzabile|Schedule serializzabile]]
	- [[#Schedule serializzabile#Esempio #1|Esempio #1]]
		- [[#Esempio #1#I possibili schedule seriali|I possibili schedule seriali]]
	- [[#Schedule serializzabile#Esempio #2|Esempio #2]]
- [[#Testare la serializzabilità|Testare la serializzabilità]]
	- [[#Testare la serializzabilità#Passo 1|Passo 1]]
		- [[#Passo 1#Esempio|Esempio]]
	- [[#Testare la serializzabilità#Passo 2|Passo 2]]
		- [[#Passo 2#Esempio|Esempio]]
- [[#Teorema (correttezza dell’algoritmo del grafo di serializzazione)|Teorema (correttezza dell’algoritmo del grafo di serializzazione)]]
	- [[#Teorema (correttezza dell’algoritmo del grafo di serializzazione)#Esempio|Esempio]]
- [[#Protocollo di locking a due fasi|Protocollo di locking a due fasi]]
- [[#Teorema sul lock a due fasi|Teorema sul lock a due fasi]]
---
## Introduction
Per **lock** si intende il **privilegio di accesso** ad un **singolo item** realizzato mediante una variabile associata all’item (variabile lucchetto) il cui valore descrive lo stato dell’item rispetto alle operazioni che possono essere effettuate su di esso (ogni item può essere locked o unlocked)

Nella sua forma più semplice, un lock:
- viene **richiesto** da una transazione mediante un’operazione di *locking* tramite la quale se il valore della variabile è unlocked la transazione può accedere all’item e alla variabile viene assegnato il valore locked
- viene **rilasciato** da una transazione mediante un’operazione di *unlocking* che assegna alla variabile il valore unlocked

Quindi il locking agisce come **primitiva di sincronizzazione**, cioè se una transazione richiede un lock su un item su cui un’altra transazione mantiene un lock, la transazione non può procedere finché il lock non viene rilasciato dalla prima transazione
Fra l’esecuzione di un’operazione di locking su un certo item $X$ e l’esecuzione di un’operazione di unlocking su $X$ la transazione **mantiene un lock su $X$**

---
## Schedule legale
Uno schedule è detto **legale** se una transazione effettua un **locking ogni volta che deve scrivere/leggere** un item e se ciascuna transazione **rilascia ogni lock** che ha ottenuto

---
## Lock binario
Un lock **binario** può assumere solo due valori: **locked** e **unlocked**

Le transazioni fanno uso di due operazioni:
- $lock(X)$ → per richiedere l’accesso all’item $X$
- $unlock(X)$ → per rilasciare l’item $X$ consentendone l’accesso ad altre transazioni

L’insieme degli item letti e quello degli item scritti da una transazione coincidono

Il lock binario permette di **risolvere il problema del lost update** (non dirty data né aggregato non corretto)
### Esempio
Risolviamo il primo dei problemi visti, cioè il lost update
![[Screenshot 2024-12-18 alle 21.22.35.png|300]]
![[Pasted image 20241218212457.png|200]]

Riscriviamo le transazioni utilizzando le primitive del lock binario
![[Pasted image 20241218223002.png|300]]

Vediamo ora uno schedule legale di $T_{1}$ e $T_{2}$ che risolve il problema del lost update
![[Pasted image 20241218223108.png|250]]

---
## Modello per le transazioni
![[Pasted image 20241218223305.png|center|180]]

Una transazione è una **sequenza di operazioni di lock e unlock**:
- ogni $lock(X)$ implica la **lettura** di $X$
- ogni $unlock(X)$ implica la **scrittura** di $X$

![[Pasted image 20241218223446.png|center|220]]

In corrispondenza di una scrittura viene associato un nuovo valore coinvolto che viene calcolato da una **funzione** che è associata in modo **univoco** ad ogni coppia lock-unlock e che ha come **argomenti tutti gli item letti** (locked) **dalla transazione prima dell’operazione di unlock** (perché magari i loro valori hanno contribuito all’aggiornamento dell’item corrente)

---
## Equivalenza
Due schedule sono **equivalenti** se le formule che danno i valori finali per ciascun item sono le stesse 

>[!warning] Le formule devono essere uguali per tutti gli item

>[!info]
>Vedremo che la proprietà di equivalenza degli schedule dipende dal protocollo di locking usato
>Adottiamo un **modello delle transazioni** per poter astrarre delle specifiche operazioni che si basa su quelle rilevanti per valutare le sequenze degli accessi, cioè in questo caso lock e unlock

---
## Schedule serializzabile
Uno schedule è **serializzabile** se è equivalente ad uno schedule seriale (basta trovarne uno)

### Esempio #1
Consideriamo le due transazioni
![[Pasted image 20241218224919.png|500]]
e lo schedule
![[Pasted image 20241218225014.png|550]]
Considerando $X_{0}$ il valore iniziale di $X$ e $Y_{0}$ il valore iniziale di $Y$ allora $f_{4}(f_{1}(X_{0}),Y_{0})$ è il valore finale di $X$

#### I possibili schedule seriali
Consideriamo lo schedule seriale $T_{1}$, $T_{2}$
![[Screenshot 2024-12-18 alle 22.54.17.png|550]]
Il valore finale di $X$ è $f_{4}(f_{1}(X_{0}),f_{2}(X_{0},Y_{0}))$

Consideriamo lo schedule seriale $T_{2}$, $T_{1}$
![[Screenshot 2024-12-18 alle 22.57.35.png|550]]
Il valore finale di $X$ è $f_{1}(f_{4}(X_{0},Y_{0}))$

Pertanto lo schedule **non è serializzabile** in quanto produce per $X$ un valore finale ($f_{4}(f_{1}(X_{0}),Y_{0})$) diverso sia da quello prodotto dallo schedule seriale $T_{1}$, $T_{2}$, sia da quello prodotto dallo schedule seriale $T_{2}$, $T_{1}$
Vale lo stesso anche per $Y$

>[!info] Osservazione
>Basta che le formule siano diverse anche per un solo item per concludere che gli schedule non sono equivalenti. Quindi per verificare che uno schedule non è serializzabile, possiamo fermarci appena troviamo un item le cui formule finali sono diverse da quelle di ogni schedule seriale
>
>Per verificare che uno schedule è serializzabile occorre verificare che le formule finali di tutti gli item coincidono con quelle di uno (stesso) schedule seriale

### Esempio #2
Consideriamo le due transazioni
![[Pasted image 20241218230404.png|500]]
e lo schedule
![[Pasted image 20241218230439.png|550]]

Consideriamo lo schedule seriale $T_{1}$, $T_{2}$
![[Screenshot 2024-12-18 alle 23.05.19.png|550]]

In questo caso lo schedule è serializzabile in quando produce sia per $X$ che per $Y$ gli stessi valori finali prodotti dallo schedule seriale $T_{1}$, $T_{2}$

---
## Testare la serializzabilità
Per testare la serializzabilità di uno schedule utilizzo il seguente algoritmo

Dato uno schedule $S$

### Passo 1
Crea un grafo diretto $G$ (*grafo di serializzazione*) in cui:
- nodi → transazioni
- archi → $T_{i}\longrightarrow T_{j}$ (con etichetta $X$) se in $S$ si ha che $T_{i}$ esegue un $unlock(X)$ e $T_{j}$ esegue il successivo $lock(X)$

>[!warning]
>Non un successivo ma **il** successivo, cioè $T_{j}$ è la prima transazione che effettua il lock di $X$ dopo che $T_{i}$ ha effettuato l’unlock, anche se le due operazioni sono di seguito

>[!hint]
>Per avere un ciclo bisogna avere archi nella stessa direzione

#### Esempio
![[Screenshot 2024-12-19 alle 21.43.12.png|350]]
Questo rappresenta il più piccolo gruppo ciclico di due transazioni

![[Screenshot 2024-12-19 alle 21.48.39.png|350]]
Questo rappresenta il più piccolo gruppo aciclico di due transazioni

### Passo 2
Se $G$ ha un ciclo allora $S$ non è serializzabile, altrimenti applicando a $G$ l’**ordinamento topologico** si ottiene uno schedule seriale $S'$ equivalente ad $S$
Per ottenere l’ordinamento topologico è necessario eliminare ricorsivamente un nodo che non ha archi entranti, insieme ai suoi archi uscenti

#### Esempio
In questo esempio i possibili punti di partenza sono $T_{1}$, $T_{4}$, $T_{7}$ quindi ho almeno $3$ possibili schedule seriali
![[IMG_56D528C66EEF-1.jpeg|center|350]]


---
## Teorema (correttezza dell’algoritmo del grafo di serializzazione)
Uno schedule $S$ è serializzabile se e solo se il suo grafo di serializzazione è **aciclico**

### Esempio
Prendiamo questo schedule di $5$ transazioni
![[Pasted image 20241219221140.png|600]]

Applichiamo l’algoritmo e segniamo sulla tabella le relazioni tra le transazioni che produrranno archi nel grafo
![[Pasted image 20241219221233.png|600]]

![[Pasted image 20241219221335.png|350]]
Il grafo presenta il ciclo $T_{1}-T_{2}-T_{3}-T_{4}$, possiamo quindi concludere che lo schedule non è serializzabile

>[!warning]
>$T_{1}-T_{2}-T_{5}$ e $T_{2}-T_{5}-T_{3}$ non sono cicli in quanto i sentieri delle frecce non descrivono cicli, mentre $T_{1}-T_{5}-T_{3}-T_{4}-T_{1}$ lo è

---
## Protocollo di locking a due fasi
Una transazione obbedisce al protocollo di **locking a due fasi** se prima effettua tutte le operazioni di lock (*fase di locking*) e poi tutte le operazioni di unlock (*fase di unlocking*)

>[!warning]
>Da non confondere con il lock a due valori. Il fatto di essere a due fasi è una caratteristica in più ma ci sono protocolli a due fasi e tre valori

**Risolve il problema dell’aggregato non corretto** (non dirty data)

---
## Teorema sul lock a due fasi
Sia $T$ un insieme di transazioni. Se ogni transazione in $T$ è a due fasi allora ogni schedule di $T$ è serializzabile

>[!info] Dimostrazioni
>Supponiamo per assurdo che ogni transazione in $S$ è a due fasi ma nel grafo di serializzazione c’è un ciclo
>![[Immagine JPEG-417D-9810-A6-0.jpeg|400]]
>Avendo uno schedule nel corso della sua esecuzione ha eseguito prima una $unlock(X_{2})$ e poi una $lock(X_{1})$ e dunque lo schema non è a due fasi **CONTRADDIZIONE**
>$\begin{flalign}&& \blacksquare\end{flalign}$

Bisogna però ricordare che questo teorema non implica il contrario. Possono quindi esistere schedule non a due fasi ma serializzabili
![[Pasted image 20241219223158.png|500]]

>[!hint]
>Tutti i protocolli di lock a due fasi (a prescindere dal numero di valori di lock) risolvono il problema dell’aggregato non corretto

