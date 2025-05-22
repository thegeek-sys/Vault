---
Created: 2025-04-11
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
---
## Index
- [[#Pila di protocolli internet|Pila di protocolli internet]]
- [[#Funzioni chiave a livello di rete|Funzioni chiave a livello di rete]]
	- [[#Funzioni chiave a livello di rete#Routing e forwarding|Routing e forwarding]]
- [[#Switch e router|Switch e router]]
	- [[#Switch e router#Link-layer switch|Link-layer switch]]
	- [[#Switch e router#Router|Router]]
- [[#Switching|Switching]]
- [[#Rete a circuito virtuale|Rete a circuito virtuale]]
	- [[#Rete a circuito virtuale#Implementazioni|Implementazioni]]
	- [[#Rete a circuito virtuale#Tabella di inoltro|Tabella di inoltro]]
- [[#Reti a datagramma|Reti a datagramma]]
	- [[#Reti a datagramma#Processo di inoltro|Processo di inoltro]]
	- [[#Reti a datagramma#Tabella di inoltro|Tabella di inoltro]]
		- [[#Tabella di inoltro#Confronta un prefisso dell’indirizzo|Confronta un prefisso dell’indirizzo]]
- [[#Funzioni del router|Funzioni del router]]
	- [[#Funzioni del router#Architettura del router|Architettura del router]]
	- [[#Funzioni del router#Porte d’ingresso|Porte d’ingresso]]
	- [[#Funzioni del router#Porte d’uscita|Porte d’uscita]]
- [[#Ricerca nella tabella di inoltro|Ricerca nella tabella di inoltro]]
- [[#Tecniche di switching (commutazione)|Tecniche di switching (commutazione)]]
	- [[#Tecniche di switching (commutazione)#Commutazione in memoria|Commutazione in memoria]]
	- [[#Tecniche di switching (commutazione)#Commutazione tramite bus|Commutazione tramite bus]]
	- [[#Tecniche di switching (commutazione)#Commutazione attraverso rete di interconnessione|Commutazione attraverso rete di interconnessione]]
- [[#Dove si verifica l’accodamento?|Dove si verifica l’accodamento?]]
	- [[#Dove si verifica l’accodamento?#Accodamento su porte di ingresso|Accodamento su porte di ingresso]]
	- [[#Dove si verifica l’accodamento?#Accodamento sulle porte di uscita|Accodamento sulle porte di uscita]]
- [[#Protocolli del livello di rete|Protocolli del livello di rete]]
---
## Pila di protocolli internet
- **applicazione** → di supporto alle applicazioni di rete
	- FTP, SMTP, HTTP
- **trasporto** → trasferimento dei messaggi a livello di applicazione tra il modulo client e server di un’applicazione
	- TCP, UDP
- **rete** → instradamento dei datagrammi dall’origine al destinatario
	- IP, protocolli di instradamento
- **link** → instradamento dei datagrammi attraverso una serie di commutatori di pacchetto
	- PPP, Ethernet
- **fisico** → trasferimento dei singoli bit

>[!example] Esempio
>![[Pasted image 20250411094148.png|center|550]]
>
>- livello di trasporto → comunicazione tra processi
>- livello di rete → comunicazione tra host
>
>Il livello di rete di $\text{H1}$ prende i segmenti dal livello di trasporto, li incapsula in un datagramma e li tramette al router più vicino. Il livello di rete di $\text{H2}$ riceve i datagrammi da $\text{R7}$, estrae i segmenti e li consegna al livello di trasporto
>Il livello di rete dei nodi intermedi inoltra verso il prossimo router

---
## Funzioni chiave a livello di rete
Il livello svolge fondamentalmente due funzioni:
- **instradamento** (*routing*)
- **inoltro** (*forwarding*)

Con l’**instradamento** si determina il percorso seguito dai pacchetti dall’origine alla destinazione (crea i percorsi). Con l’**inoltro** si trasferiscono i pacchetti dall’input di un router all’output del router appropriato (utilizza i percorsi creati dal routing)

>[!hint]
>Gli algoritmi di routing creano le tabelle di routing che vengono usate per il forwarding

### Routing e forwarding
In sintesi si può dire quindi che il routing algorithm crea la **forwarding table** (determina i valori inseriti nella tabella), ovvero una tabella che specifica quale collegamento di uscita bisogna prendere per raggiungere la destinazione

>[!warning]
>Ogni router ha la propria forwarding table

![[Pasted image 20250411095158.png|450]]

---
## Switch e router
Il **packet switch** (commutatore di pacchetto) è un dispositivo che si occupa del trasferimento dall’interfaccia di ingresso a quella di uscita in base al valore del campo dell’intestazione del pacchetto

![[Pasted image 20250411100535.png]]

### Link-layer switch
I **link-layer switch** (commutatore a livello di commutatore) stabiliscono l’inoltro in relazione al valore nel campo nel livello di collegamento (livello 2)
Hanno dunque il compito di instradare i pacchetto al livello 2 (collegamento), e viene utilizzato per **collegare singoli computer all’interno di una rete LAN**

![[Pasted image 20250411100811.png]]

### Router
I **router** stabiliscono l’inoltro in base al valore del campo nel livello di rete (livello 3)
Hanno dunque il compito di instradare i pacchetti al livello 3 (rete), inoltrando un pacchetto che arriva su uno dei suoi collegamenti di comunicazione ad uno dei sui collegamenti di comunicazione (*next hop*)

![[Pasted image 20250411101039.png]]

---
## Switching
Esistono due **tipi di rete**, a:
- **circuito virtuale** (servizio orientato alla connessione) → prima che i datagrammi fluiscano, i due sistemi terminali e i router intermedi stabiliscano una connessione virtuale
- **datagramma** (servizio senza connessione) → ogni datagramma viaggia indipendentemente dagli altri

La rete gestisce il percorso dei pacchetti (modelli di comportamento della rete)

Per ogni rete però esistono diverse **tecniche di commutazione**:
- commutazione in memoria
- commutazione tramite bus
- commutazione attraverso rete d’interconnessione

Le tecniche di commutazione gestiscono come un dispositivo inoltra i pacchetti al suo interno (meccanismo interno all’apparato)


---
## Rete a circuito virtuale
Il pacchetto di un circuito virtuale ha un numero **VC** (etichetta di circuito) nella propria intestazione.
Un circuito virtuale può avere un numero VC diverso su ogni collegamento, infatti ogni router sostituisce il numero VC con un nuovo numero

![[Pasted image 20250411101704.png]]

>[!info] ATM (Asynchronous transfer mode)
>La ATC è la prima rete orientata alla connessione, progettata nei primi anni 90.
>Il suo scopo era quello di unificare voce, dati, televisione via cavo, …
>
>Attualmente viene usata nella rete telefonica per trasportare (internamente) pacchetti IP. Quando una connessione è stabilita, ciascuna parte può inviare dati (suddivisi in celle di 53 bytes)
### Implementazioni
Un circuito virtuale consiste in:
1. un percorso tra gli host origine e destinazione
2. numeri VC, uno per ciascun collegamento
3. righe nella tabella d’inoltro in ciascun router

Il pacchetto di un circuito virtuale ha un numero VC nella propria intestazione che rappresenta un’etichetta di flusso. Questo numero cambia su tutti i collegamenti lungo il percorso (il nuovo numero viene rilevato dalla tabella di inoltro)

### Tabella di inoltro
![[Pasted image 20250411102430.png]]

>[!warning] I router mantengono le informazioni sullo stato delle connessioni
>Aggiungono alla tabella d’inoltro una nuova riga ogni volta che stabiliscono una nuova connessione (la cancellano quando la connessione viene rilasciata)

---
## Reti a datagramma
Internet è una **rete a datagramma** (*packet switched*). In questo tipo di rete l’impostazione della chiamata non avviene a livello di rete e i router non conservano informazioni sullo stato dei circuiti virtuali (non c’è il concetto di “connessione” a livello di rete)

I pacchetti vengono inoltrati utilizzano l’indirizzo dell’host destinatario, passando attraverso una serie di router che utilizzano gli indirizzi di destinazione per inviarli (possono intraprendere percorsi diversi)

![[Pasted image 20250411103729.png]]

### Processo di inoltro
![[Pasted image 20250411104650.png]]

### Tabella di inoltro
![[Pasted image 20250411104816.png]]

#### Confronta un prefisso dell’indirizzo
Quando si verificano corrispondenze multiple si prende la corrispondenza a **prefisso più lungo**, in cui viene determinata la corrispondenza più lunga all’interno della tabella e si inoltrano i pacchetti sull’interfaccia corrispondente, garantendo la **continuità** degli indirizzi

>[!example] Esempio
>![[Pasted image 20250411105035.png|400]]
>
>- `11001000 00010111 00010110 10100001` → $0$
>- `11001000 00010111 00011000 10101010` → $1$

---
## Funzioni del router
![[Pasted image 20250411105336.png]]

### Architettura del router
![[Pasted image 20250411105411.png]]

### Porte d’ingresso
![[Pasted image 20250411105557.png]]

### Porte d’uscita
![[Pasted image 20250411111935.png]]

- **Funzionalità di accodamento** → quando la struttura di commutazione consegna pacchetti alla porta d’uscita a una frequenza che supera quella del collegamento uscente
- **Schedulatore di pacchetti** → stabilisce in quale ordine trasmettere i pacchetti accodati

---
## Ricerca nella tabella di inoltro
La ricerca nella tabella di inoltro deve essere veloce (possibilmente con lo stesso tasso della linea) per evitare accodamento, per questo motivo è implementata in una **struttura ad albero**.

Ogni livello dell’albero corrisponde ad un bit dell’indirizzo di destinazione, dunque per cercare un indirizzo si comincia dalla radice dell’albero (se $0$ allora sottoalbero di sinistra, se $1$ allora sottoalbero di destra), garantendo una ricerca in $N$ passi dove $N$ è il numero di bit nell’indirizzo

---
## Tecniche di switching (commutazione)
![[Pasted image 20250411110957.png]]

### Commutazione in memoria
Riguarda la prima generazione di router, in cui questi erano dei tradizionali calcolatori e la commutazione era effettuata sotto il controllo diretto della CPU.
Il pacchetto veniva copiato nella memoria del processore e in seguito veniva trasferito dalle porte di ingresso a quelle di uscita

![[Pasted image 20250411111210.png|600]]

### Commutazione tramite bus
In questo caso le porte di ingresso trasferiscono un pacchetto direttamente alle porte di uscita su un bus condiviso, senza l’intervento del processore di instradamento, ma questo permette di trasferire un solo pacchetto alla volta.
Infatti i pacchetti che arrivano e trovano il bus occupato vengono accodati alla porta di ingresso, limitando la larghezza di banda a quella del bus

![[Pasted image 20250411111459.png|350]]

### Commutazione attraverso rete di interconnessione
Questo tipo di commutazione supera il limite di banda si un singolo bus condiviso.
La **crossbar switch** infatti è una rete di interconnessione che consiste di $2n$ bus che collegano $n$ porte d’ingresso a $n$ porte di uscita.

Attualmente si tende a frammentare dei pacchetti IP a lunghezza variabile in celle di lunghezza fissa, per poi essere riassemblati nella porta di uscita

![[Pasted image 20250411111802.png|280]]

---
## Dove si verifica l’accodamento?
L’accodamento si verifica sia nelle porte di uscita che in quelle di ingresso

>[!info] Velocità di commutazione
>Frequenza alla quale tale un router può trasferire i pacchetti dalle porte di ingresso a quelle di uscita

>[!question] Quale deve essere la capacità dei buffer?
>Per diversi anni si è seguita la regola definita in RFC 3439: la quantità di buffering dovrebbe essere uguale a una media del tempo di andata e ritorno (RTT ad esempio $250 \text{ ms}$) per la capacità del collegamento $C$
>
>Le attuali raccomandazioni invece dicono che la quantità di buffering necessaria per $N$ flussi TCP è:
>$$\frac{\text{RTT}\cdot \text{C}}{\sqrt{ n }}$$
### Accodamento su porte di ingresso
Si ha accodamento su porte di ingresso quando la struttura di commutazione ha una velocità inferiore a quella delle porte di ingresso (per non avere accodamento la velocità di commutazione dovrebbe essere $n\cdot \text{velocità della linea di ingresso}$)

Oltre a ciò, anche quando si ha un **blocco in testa alla fila** (*HOL - head-of-line blocking*), ovvero quando un pacchetto nella coda di ingresso deve attendere il trasferimento (anche se la propria destinazione è libera) in quanto risulta bloccato da un altro pacchetto in testa alla fila

>[!hint]
>Se le code diventano troppo lunghe, i buffer si possono saturare e quindi causare una perdita di pacchetti

![[Pasted image 20250411112815.png|500]]

### Accodamento sulle porte di uscita
Se la struttura di commutazione non è sufficientemente rapida nel trasferire i pacchetti, si può verificare un accodamento sulle porte di uscita

Potrebbe avvenire anche nel caso in cui troppi pacchetti vanno sulla stessa porta di uscita

![[Pasted image 20250411113034.png|450]]

>[!hint]
>Se le code diventano troppo lunghe, i buffer si possono saturare e quindi causare una perdita di pacchetti

---
## Protocolli del livello di rete

![[Pasted image 20250411113431.png|center|600]]

- **IP** → *Internet Protocol* v4 (anche v6)
- **IGMP** → *Internet Group Management Protocol* (multicasting)
- **ICMP** → *Internet Control Message Protocol* (gestione errori)
- **ARP** → *Address Resolution Protocol* (associazione indirizzo IP – ind. collegamento)
