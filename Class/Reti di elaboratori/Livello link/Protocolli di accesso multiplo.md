---
Created: 2025-05-10
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Protocolli a suddivisione del canale|Protocolli a suddivisione del canale]]
	- [[#Protocolli a suddivisione del canale#TDMA|TDMA]]
	- [[#Protocolli a suddivisione del canale#FDMA|FDMA]]
- [[#Protocolli ad accesso casuale|Protocolli ad accesso casuale]]
	- [[#Protocolli ad accesso casuale#ALOHA|ALOHA]]
	- [[#Protocolli ad accesso casuale#ALOHA puro|ALOHA puro]]
		- [[#ALOHA puro#Timeout e back-off|Timeout e back-off]]
		- [[#ALOHA puro#Compromessi|Compromessi]]
		- [[#ALOHA puro#Efficienza|Efficienza]]
	- [[#Protocolli ad accesso casuale#Slotted ALOHA|Slotted ALOHA]]
		- [[#Slotted ALOHA#Pro e contro|Pro e contro]]
		- [[#Slotted ALOHA#Efficienza|Efficienza]]
	- [[#Protocolli ad accesso casuale#CSMA|CSMA]]
	- [[#Protocolli ad accesso casuale#CSMA/CD|CSMA/CD]]
		- [[#CSMA/CD#Metodi di persistenza|Metodi di persistenza]]
			- [[#Metodi di persistenza#Non persistente|Non persistente]]
			- [[#Metodi di persistenza#1 persistente|1 persistente]]
			- [[#Metodi di persistenza#p persistente|p persistente]]
		- [[#CSMA/CD#Efficienza|Efficienza]]
- [[#Protocolli MAC a rotazione|Protocolli MAC a rotazione]]
	- [[#Protocolli MAC a rotazione#Protocollo polling|Protocollo polling]]
	- [[#Protocolli MAC a rotazione#Protocollo token-passing|Protocollo token-passing]]
---
## Introduction
Esistono due tipi di collegamenti di rete:
- **collegamento punto-punto**, impiegato per
	- connessioni telefoniche
	- collegamenti punto-punto tra Ethernet e host
	- point-to-point protocol (PPP) del DLC
- **collegamento broadcast** (cavo o canale condiviso), impiegato per
	- Ethernet tradizionale
	- Wireless LAN 802.11

In una connessione a un canale broadcast condiviso, centinaia o anche migliaia di nodi possono comunicare direttamente su un canale broadcast e si genera una collisione quando i nodi ricevono due o più frame contemporaneamente

Con i protocolli di accesso multiplo l’obiettivo è quello di evitare caos e realizzare una condivisione. I protocolli dunque fissano le modalità con cui i nodi regolano le loro trasmissioni sul canale condiviso

>[!warning]
>La comunicazione relativa al canale condiviso deve utilizzare lo stesso canale (non c’è un canale “out-of-bound“ per la coordinazione)

>[!info] Protocolli di accesso multiplo ideali
>Canale broadcast con velocità di $R$ bit al secondo:
>1. quando un nodo deve inviare dati, questo dispone di un tasso trasmissivo pari a $R$ bps
>2. quando $M$ nodi devono inviare dati, questi dispongono di un tasso trasmissivo pari a $\frac{R}{M}$ bps
>3. il protocollo è decentralizzato
>	- non ci sono nodi master
>	- non c’è sincronizzazione dei clock

I protocolli di accesso multiplo si possono classificare in una di queste tre categorie:
- **protocolli a suddivisione del canale** (*channel partitioning*) → suddivide un canale in “parti più piccole” (slot di temo, frequenza, codice) e li colloca presso un nodo per utilizzo esclusivo (non si hanno collisioni)
- **protocolli ad accesso casuale** (*random access*) → i canali non vengono divisi e si può verificare una collisione; i nodi coinvolti ritrasmettono ripetutamente i pacchetti
- **protocolli a rotazione** (*”taking-turn”*) → ciascun nodo ha il suo turno di trasmissione, ma i nodi che hanno molto da trasmettere possono avere turni più lunghi

![[Pasted image 20250510122402.png]]

---
## Protocolli a suddivisione del canale
### TDMA
Il **time division multiple access** (*TDMA*) utilizza dei turni per accedere al canale (ogni nodo ha un turno assegnato) e suddivide il canale condiviso in intervalli di tempo
Gli slot non usati rimangono inattivi

>[!example]
>Gli slot $1$, $3$ e $4$ hanno un pacchetto, $2$, $5$ e $6$ sono inattivi
>![[Pasted image 20250510122618.png]]

Il tasso trasmissivo risulta essere $\frac{R}{N}$ bps e non è flessibile rispetto a variazioni nel numero di nodi

### FDMA
Il **frequency division multiple access** (*FDMA*) suddivide il canale in bande di frequenza e a ciascuna stazione è assegnata una banda di frequenza prefissata

>[!example]
>Le bande $1$, $3$ e $4$ hanno un pacchetto, $2$, $5$ e $6$ sono inattive
>![[Pasted image 20250510122816.png]]

### CDMA
Nel **code division multiple access** (*CDMA*) si ha un solo canale che occupa l’intera ampiezza di banda (non c’è divisione di frequenze) e tutte le stazioni possono inviare contemporaneamente (non si ha divisione di tempo)

>[!example]
>Assumiamo di avere 4 stazioni connesse sullo stesso canale
>I dati spediti sono $d_{1},d_{2},d_{3},d_{4}$, i codici assegnati sono $c_{1},c_{2},c_{3},c_{4}$
>
>Ogni stazione moltiplica i propri dati per il proprio codice e trasmette
>![[Pasted image 20250522000215.png]]

#### Proprietà dei codici
I codici godono di diverse proprietà, in particolare:
- se moltiplichiamo ogni codice per un altro otteniamo 0
- se moltiplichiamo ogni codice per sé stesso otteniamo il numero delle stazioni (4)

Dunque qualsiasi stazione voglia ricevere dati da una delle altre tre stazioni moltiplica i dati ricevuti per il codice del mittente e divide per il numero delle stazioni

>[!example] Stazione 2 vuole ricevere dalla stazione 1
>$$\begin{align}\text{dati}=&\frac{(d_{1}\cdot c_{1}+d_{2}\cdot c_{2}+d_{3}\cdot c_{3}+d_{4}\cdot c_{4})\cdot c_{1}}{4}=\\=&\frac{d_{1}\cdot c_{1}\cdot c_{1}+d_{2}\cdot c_{2}\cdot c_{1}+d_{3}\cdot c_{3}\cdot c_{1}+d_{4}\cdot c_{4}\cdot c_{1}}{4}=\frac{4\cdot d_{1}}{4}=d_{1}\end{align}$$

#### Sequenze ortogonali
Il CDMA si basa sulla teoria della codifica. Ad ogni stazione viene assegnato un codice che è una sequenza di numeri chiamati **chip** (*sequenze ortogonali*)

Le sequenze ortogonali godono di svariate proprietà:
- ogni sequenza è composta da $N$ elementi (stazioni), dove $N$ deve essere una potenza di 2
- se moltiplichiamo una sequenza per un numero, ogni elemento della sequenza viene moltiplicato per tale numero 
	$$2\cdot[+1+1-1-1]=[+2+2-2-2]$$
- se moltiplichiamo due sequenze uguali e sommiamo i risultati otteniamo $N$
	$$[+1+1-1-1]\cdot[+1+1-1-1]=1+1+1+1=4$$
- se moltiplichiamo due sequenze diverse e sommiamo i risultati otteniamo 0
	$$[+1+1-1-1]\cdot[+1+1+1+1]=1+1-1-1=0$$
- sommare due sequenze significa sommare gli elementi corrispondenti
	$$[+1+1+1+1]+[+1+1+1+1]=[+2+2\;\,0\;0\,]$$

#### Rappresentazione dei dati
Regole per la codifica:
![[Pasted image 20250522001621.png|450]]

>[!example] La stazione 3 ascolta la 2
>![[Pasted image 20250522001722.png]]
>$$[-1-1-3+1]\cdot[+1-1+1-1]=-4\to\frac{-4}{4}=-1\to \text{bit }0$$

#### Generazione sequenze di chip
Per generare sequenze di chip usiamo una tabella di Walsh (matrice quadrata). Nella tabella di Walsh ogni riga è una sequenza di chip

$W_{1}$ indica una sequenza con un chip solo (con una riga e una colonna) e può assumere valore $+1$ o $-1$ (a scelta). Conoscendo $W_{N}$ possiamo creare $W_{2N}$ nel seguente modo:
$$
W_{1}=\begin{bmatrix}
+1
\end{bmatrix}\;\, W_{2N}=
\begin{bmatrix}
W_{N}&W_{N} \\
W_{N}&\overline{W_{N}}
\end{bmatrix}
$$

$$
W_{2}=
\begin{bmatrix}
\colorbox{darkjunglegreen}{+1}+1&+1 \\
+1&+1
\end{bmatrix}
$$
>[!example] Esempio
>

---
## Protocolli ad accesso casuale
Nei protocolli ad accesso casuale nessuna stazione ha il controllo sulle altre, infatti ogni volta che una stazione ha dei dati da inviare usa una procedura definita dal protocollo per decidere se spedire o meno

Per **accesso casuale** si intende che non c’è un tempo programmato nel quale la stazione deve trasmettere e che nessuna regola specifica quale sarà la prossima stazione a trasmettere

Le stazioni competono l’una con l’altra per accedere al mezzo trasmissivo (*contesa del canale*) e se ci sono due o più nodi trasmettenti si verifica una **collisione**

Inoltre il protocollo ad accesso casuale definisce come rilevare un’eventuale collisione e come ritrasmettere se si è verificata una collisione

>[!example] Esempi di protocolli
>- ALOHA
>- slotted ALOHA
>- CSMA
>- CSMA/CD
>- CSMA/CA

>[!info] Efficienza
>L’efficienza è definita come la frazione di slot vincenti in presenza di un elevato numero $N$ di nodi attivi, che hanno sempre un elevato numero di pacchetti da spedire

### ALOHA
L’**ALOHA** è stato il primo metodo di accesso casuale che è stato proposto in letteratura ed è stato sviluppato all’Università delle Hawaii nei primi anni 70
E’ stato ideato per mettere in comunicazione gli atolli mediante una LAN radio (wireless) ma può essere utilizzato su qualsiasi mezzo trasmissivo

Essendo un protocollo ad accesso casuale possono verificarsi collisioni

### ALOHA puro
Nell’ALOHA puro ogni stazione può inviare un frame tutte le volte che ha dati da inviare e il ricevente invia un ACK per notificare la corretta ricezione del frame
Se il mittente non riceve una ACK entro un *timeout* deve ritrasmettere

Se due stazioni ritrasmettono contemporaneamente si ha di nuovo una collisione, allora si attende un tempo random (**back-off**) prima di effettuare la ritrasmissione (è proprio la casualità del back-off che aiuta ad evitare altre collisioni)

Dopo un numero massimo di tentativi $K_{\text{max}}$ una stazione interrompe i suoi tentativi e riprova più tardi

>[!example]
>![[Pasted image 20250510125012.png]]

>[!warning]
>La durata della connessione è variabile, può essere anche di un solo bit

#### Timeout e back-off

>[!question] Per quanto tempo si aspetta l’ACK?
>Il periodo di timeout equivale al massimo ritardo di propagazione di round-trip (andata del frame e ritorno dell’ACK) tra le due stazioni più lontane ($2\times T_{p}$)

>[!question] Quanto si aspetta prima di ritrasmettere
>Il tempo di back-off $T_{B}$ è un valore scelto casualmente che dipende dal numero $K$ di trasmissioni fallite
>
>$$ \begin{align}\text{Backoff time}=R\cdot T_{fr}\qquad\text{dove}\quad &R\in[0,2^k-1] \\ &K=\#\text{tentativi} \\ &T_{fr}=\text{tempo per inviare un frame} \\ &K_{\text{max}}=15 \end{align} $$
>
>>[!example] Calcolo back-off
>>Le stazioni di una rete wireless ALOHA sono a una distanza massima di $600\text{ km}$. Supponendo che i segnali si propaghino a $3\times 10^8\text{ m/s}$, troviamo:
>>$$T_{p}=\frac{600\cdot 10^3}{3\cdot 10^8}=2\text{ ms}$$
>>
>>Per $K=2$ l’intervallo di $R$ è $\{0,1,2,3\}$. Ciò significa che $T_{B}=R\cdot T_{Fr}$ può essere $0,2,4,6 \text{ ms}$ sulla base del risultato della variabile casuale $R$

#### Compromessi
Nonostante tutto ciò questo protocollo ha:
- elevate probabilità di collisione
- tempo di vulnerabilità → l’intervallo di tempo nel quale il frame è a rischio di collisioni
	- il frame trasmesso a $t$ si sovrappone con la trasmissione di qualsiasi altro frame inviato in $[t-1,t+1]$, dunque il tempo di vulnerabilità risulta essere $2T_{fr}$

![[Pasted image 20250510171946.png|500]]

#### Efficienza
Assumiamo che tutti i frame hanno la stessa dimensione e ogni nodo ha sempre un frame da trasmettere. In ogni istante di tempo, $p$ è la probabilità che un nodo trasmetta un frame, $1-p$ che non trasmetta

Supponendo che un nodo inizi a trasmettere al tempo $t_{0}$, perché la trasmissione vada a buon fine, nessun altro nodo deve aver iniziato una trasmissione nel tempo $[t_{0}-1,t_{0}]$. Tale probabilità è data da $(1-p)^{N-1}$

Allo stesso modo nessun nodo deve iniziare a trasmettere nel tempo $[t_{0},t_{0}+1]$, e la probabilità di questo evento è ancora $(1-p)^{N-1}$

La probabilità che un nodo trasmetta con successo è dunque $p(1-p)^{2(N-1)}$
Studiando il valore di $p$ che massimizza la probabilità di successo per $N$ che tende a infinito si ottiene che l’efficienza massima è $\frac{1}{2}e$ ovvero $0.18$ che risulta molto bassa

$$
\begin{align}
P(\text{trasmissione c}&\text{on successo di un dato nodo})= \\
&P(\text{il nodo trasmette})\cdot P(\text{nessun altro nodo trasmette in }[t_{0}-1,t_{0}])\,\cdot \\
&P(\text{nessun altro nodo trasmette in }[t_{0},t_{0}+1])= \\
&=p\cdot(1-p)^{N-1}\cdot(1-p)^{N-1}=p\cdot(1-p)^{2(N-1)}\underset{N\to\infty}{\longrightarrow} \frac{1}{2e}=0.18
\end{align}
$$

Dunque il throughput è $0.18R$ bps

### Slotted ALOHA
Un modo per aumentare l'efficienza di ALOHA (Roberts, 1972) consiste nel dividere il tempo in intervalli discreti, ciascuno corrispondente ad un frame time ($T_{fr}$)
I nodi dunque devono essere d’accordo nel confine fra gli intervalli, e ciò può essere fatto facendo emettere da una attrezzatura speciale un breve segnale all’inizio di ogni intervallo

Assumiamo che:
- tutti i pacchetti hanno la stessa dimensione
- il tempo è suddiviso in slot; ogni slot equivale al tempo di trasmissione di un pacchetto
- i nodi iniziano la trasmissione dei pacchetti solo all’inizio degli slot
- i nodi sono sincronizzati
- se in uno slot due o più pacchetti collidono, i nodi coinvolti rilevano l’evento prima del termine dello slot

![[Pasted image 20250510175510.png]]

Dunque quando a un nodo arriva un nuovo pacchetto da spedire, il nodo attende l’inizio del prossimo slot. Se non si verifica una collisione il nodo può trasmette un nuovo pacchetto nello slot successivo, altrimenti il nodo ritrasmette con probabilità $p$ il suo pacchetto durante gli slot successivi

#### Pro e contro
**PRO**
- consente ad un singolo nodo di trasmette continuamente pacchetti alla massima velocità del canale
- il tempo di vulnerabilità si riduce ad un solo slot ($T_{fr}$)

**CONTRO**
- una certa frazione degli slot presenterà collisioni e di conseguenza andrà “sprecata”
- un’altra frazione degli slot rimane vuota, quindi inattiva

#### Efficienza
Supponiamo $N$ nodi con pacchetti da spedire, ognuno trasmette i pacchetti in uno slot con probabilità $p$. La probabilità di successo di un dato nodo è $p(1-p)^{N-1}$
Poiché ci sono $N$ nodi, la probabilità che ogni nodo abbia successo è $N\cdot p(1-p)^{N-1}$

Per un elevato numero di nodi ricaviamo il limite per $N$ che tende a infinito, ovvero $\frac{1}{e}=0.37$
Il throughput non è $R$ ma $0.37R$ bps

### CSMA
Nel **Carrier Sense Multiple Access** (*CSMA*) si pone in ascolto prima di trasmettere (*sense before transmit*). Se rileva che il canale è libero, trasmette l’intero pacchetto, se invece il canale sta già trasmettendo, il nodo aspetta un altro intervallo di tempo

Nonostante ciò le collisioni possono ancora verificarsi, infatti il ritardo di propagazione fa si che due nodi non rilevino la reciproca trasmissione (il tempo di vulnerabilità è il tempo di propagazione)

>[!note]
>La distanza e il ritardo di propagazione giocano un ruolo importante nel determinare la probabilità di collisione

![[Pasted image 20250510175832.png|320]]

### CSMA/CD
Con il **CSMA/CD** (*collision detection*) è possibile anche rilevare una collisione, infatti ascolta il canale anche durante la trasmissione. Ciò permette di rilevare la collisione in poco tempo, e viene annullata la trasmissione non appena si accorge che c’è un’altra trasmissione in corso

Rilevare la collisione però risulta essere facile nelle LAN cablate ma difficile nelle LAN wireless

>[!example]
>![[Pasted image 20250510180203.png]]
>
>- $A$ ascolta il canale e inizia la trasmissione al tempo $t_{1}$
>- $C$ al tempo $t_{2}$ ascolta il canale (non rileva ancora il prossimo bit di $A$) e quindi inizia a trasmettere
>- al tempo $t_{3}$ $C$ riceve il primo bit di $A$ e interrompe la trasmissione perché c’è collisione
>- al tempo $t_{4}$ $A$ riceve il primo bit di $C$ e interrompe la collisione perché c’è collisione

>[!question] Cosa ssuccederebbe se un mittente finisse di trasmettere un frame prima di ricevere il primo bit di un’altra stazione (che ha già iniziato a trasmettere)?
>Una stazione una volta inviato un frame non tiene una copia del frame, né controlla il mezzo trasmissivo per rilevare collisioni
>Perché il Collision Detection funzioni, il mittente deve poter rilevare la trasmissione mentre sta trasmettendo ovvero prima di inviare l’ultimo bit del frame
>
>Il tempo di trasmissione di un frame deve essere almeno due volte il tempo di propagazione $T_{p}$; quindi la prima stazione deve essere ancora in trasmissione dopo $2T_{p}$

>[!example]
>Una rete che utilizza il CSMA/CD ha un rate $10 \text{ Mbps}$. Se il tempo di propagazione massimo è $25,6 \,\mu \text{s}$, qual è la dimensione minima del frame?
>>[!done]
>>Il tempo di trasmissione del frame è
>>$$T_{fr}=2\cdot T_{p}=51.2\,\mu \text{s}$$
>>
>>Ciò significa, nel peggiore dei casi, che una stazione deve trasmettere per un periodo di $51.2\,\mu \text{s}$ per poter rilevare la collisione
>>
>>La dimensione minima del frame è quindi:
>>$$10\text{ Mbps}\cdot 51.2\,\mu\text{s}=512\text{ bit} \text{(o }64\text{ byte)}$$
>>
>>Questa è proprio la dimensione massima del frame nell’Ethernet Standard come si vedrà

#### Metodi di persistenza

>[!question] Cosa fa un nodo se trova il canale libero?
>- trasmette subito
>	- non persistente
>	- $1$-persistente
>- trasmette con probabilità $p$
>	- $p$-persistente

>[!question] Cosa fa un nodo se trova il canale occupato?
>- desiste → riascolta dopo un tempo random
>	- non persistente
>- persiste → rimane in ascolto finché il canale non si è liberato
>	- $1$-persistente
>	- $p$-persistente (usato in presenta di time slot)

##### Non persistente
- se il canale è libero trasmette immediatamente
- se il canale è occupato attende un tempo random e poi riascolta il canale (carrier sense a intervalli)
- se collisione back-off

![[Pasted image 20250510181548.png|500]]

##### 1 persistente
- se il canale è libero trasmette immediatamente ($p=1$)
- se il canale è occupato continua ad ascoltare (carrier sense continuo)
- se collisione back-off

![[Pasted image 20250510181656.png|500]]

##### p persistente
- se il canale è libero
	- trasmette con probabilità $p$
	- rimanda la trasmissione con probabilità $1-p$
- se il canale è occupato usa la procedura di back-off (attesa di un tempo random e nuovo ascolto del canale)
- se collisione back-off

![[Pasted image 20250510181827.png]]

#### Efficienza
Quando un solo nodo trasmette, il nodo può trasmettere al massimo rate (es. $10\text{ Mbps}$, $100\text{ Mbps}$, $1\text{ Gbps}$). Quando più nodi trasmettono, il rate effettivo o throughput è minore

Il throughput del CSMA/CD è maggiore sia dell’ALOHA puro che dello slotted ALOHA
Per il metodo 1 persistente (che è anche il caso del’Ethernet tradizionale ovvero $10\text{ Mbps}$) il throughput massimo è del 50%

---
## Protocolli MAC a rotazione
Protocolli MAC a suddivisione del canale:
- condividono il canale equamente ed efficientemente con carichi elevati
- inefficienti con carichi non elevati

Protocolli MAC ad accesso casuale:
- efficienti anche con carichi non elevati: un singolo nodo può utilizzare interamente il canale
- con carichi elevati si ha un eccesso di collisioni

I **protocolli a rotazione** cercano di realizzare un compromesso tra i protocolli precedenti

### Protocollo polling
![[Pasted image 20250510182501.png|center|280]]

Un nodo principale sonda “a turno” gli altri. In particolare:
- elimina le collisioni
- elimina gli slot vuoti
- ritardo di polling
- se il nodo principale (master) si guasta, l’intero canale resta inattivo

### Protocollo token-passing
![[Pasted image 20250510182648.png|center|280]]

Un messaggio di controllo (**token**) circola fra i nodi seguendo un ordine prefissato. In particolare è decentralizzato, altamente efficiente e il guasto di un nodo può mettere fuori uso l’intero canale