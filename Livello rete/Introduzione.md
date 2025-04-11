---
Created: 2025-04-11
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
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
Si hanno due approcci allo switching:
- **circuito virtuale** (servizio orientato alla connessione) → prima che i datagrammi fluiscano, i due sistemi terminali e i router intermedi stabiliscano una connessione virtuale
- **datagramma** (servizio senza connessione) → ogni datagramma viaggia indipendentemente dagli altri

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

---
## Ricerca nella tabella di inoltro
La ricerca nella tabella di inoltro deve essere veloce (possibilmente con lo stesso tasso della linea) per evitare accodamento, per questo motivo è implementata in una **struttura ad albero**.

Ogni livello dell’albero corrisponde ad un bit dell’indirizzo di destinazione, dunque per cercare un indirizzo si comincia dalla radice dell’albero (se $0$ allora sottoalbero di sinistra, se $1$ allora sottoalbero di destra), garantendo una ricerca in $N$ passi dove $N$ è il numero di bit nell’indirizzo

