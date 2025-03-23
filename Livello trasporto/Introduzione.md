---
Created: 2025-03-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
---
## Connessione logica a livello trasporto
I protocolli di trasporti forniscono la **comunicazione logica** tra processi applicativi di host differenti. Con la connessione logica gli host eseguono i processi come se fossero direttamente connessi (in realtà possono trovarsi agli antipodi del pianeta)

I protocolli di trasporto vengono eseguiti nei sistemi terminali:
- lato invio → incapsula i messaggi in **segmenti** e li passa al livello di rete
- lato ricezione → decapsula i segmenti in messaggi e li passa al livello di applicazione

![[Pasted image 20250323112619.png|550]]

---
## Relazione tra livello di trasporto e livello di rete
![[Pasted image 20250323113733.png]]

Mentre il livello di rete regola la **comunicazione tra host** (si basa sui servizi del livello di collegamento), il livello di trasporto regola la **comunicazione tra processi** (si basa sui servizi del livello di rete e li potenzia)

>[!example] Analogia con la posta ordinaria
>Una persona di un condominio invia una lettera a una persona di un altro condominio consegnandola/ricevendola a/da un portiere
>
>In questo caso:
>- i processi sono le persone
>- i messaggi delle applicazioni sono le lettere nelle buste
>- gli host sono i condomini
>- il protocollo di trasporto sono i portieri dei condomini
>- il protocollo del livello di rete è il servizio postale
>
>>[!hint]
>>I portieri svolgono il proprio lavoro localmente, non sono coinvolti nelle tappe intermedie delle lettere (così come il livello di trasporto)

---
## Indirizzamento
La maggior parte dei sistemi operativi è multiutente e multiprocesso (ci sono diversi processi client attivi e diversi processi server attivi)

![[Pasted image 20250323115546.png|center|500]]

Per stabilire una comunicazione tra i due processi è necessario un metodo per individuare:
- host locale → tramite indirizzo IP
- host remoto → tramite indirizzo IP
- processo locale → tramite numero di porta
- processo remoto → tramite numero di porta

>[!info] Indirizzi IP vs. numeri di porta
>![[Pasted image 20250323115740.png]]
>
>L’indirizzo IP + la porta forma il **socket address**

---
## Incapsulamento/decapsulamento
![[Pasted image 20250323115919.png]]

I pacchetto a livello di trasporto sono chiamati **segmenti** (*TCP*) o **datagrammi utente** (*UDP*)

---
## Multiplexing/demultiplexing
Come il servizio di trasporto da host a host fornito dal livello di rete possa diventare un servizio di trasporto da processo a processo per le applicazioni in esecuzione sugli host

![[Pasted image 20250323164548.png|500]]

>[!example] Esempio dei condomini
>I portieri effettuano un’operazione di:
>- multiplexing quando raccolgono le lettere dai condomini (mittenti) e le imbucano
>- demultiplexing quando ricevono le lettere dal postino, leggono il nome riportato su ciascuna busta e consegnano ciascuna lettera la rispettivo destinatario

### Multiplexing
Il **mutliplexing** viene utilizzato dall’host mittente per raccogliere i dati da varie socket, incapsularli con l’intestazione (utilizzata poi per il demultiplexing)

>[!example]
>Su un host ci sono due processi in esecuzione
>- $P_{1}=\text{FTP}$
>- $P_{2}=\text{HTTP}$
>
>L’host deve raccogliere i dati in uscita da queste socket e passarli al livello di rete
>![[Pasted image 20250323164824.png]]

### Demultiplexing
Il **demutliplexing** viene utilizzato dall’host ricevente per consegnare i segmenti ricevuti alla socket appropriata

>[!example]
>Su un host ci sono due processi in esecuzione
>- $P_{1}=\text{FTP}$
>- $P_{2}=\text{HTTP}$
>
>Quando il livello di trasporto dell’host riceve i dati dal livello di rete sottostante, deve indirizzare i dati a uno di questi processi. Ciò viene fatto con le informazioni all’interno dell’header
>![[Pasted image 20250323165048.png]]

### Come funziona il demultiplexing
L’host riceve i datagrammi IP (livello di rete). In ogni datagramma è presente l’indirizzo IP di origine e l’indirizzo IP di destinazione.
Ogni datagramma trasporta un segmento a livello di trasporto. Ogni segmento ha un numero di porta di origine e un numero di porta di destinazione

![[Pasted image 20250323165735.png|center|350]]

L’host usa gli indirizzi IP e i numeri di porta per inviare il segmento al processo appropriato
Il campo n° di porta è composto da $16\text{ bit}$ con valori che vanno da $0$ a $65535$ (fino a $1023$, *well-known-port number*)

>[!example]
>![[Pasted image 20250323165807.png]]

---
## API di comunicazione
Il **socket** è un’interfaccia che si trova tra il livello applicazione e livello di trasporto e che permette di farli comunicare

![[Pasted image 20250323165953.png]]

### Comunicazione tra processi
Il **socket** appare come un terminale o un file ma non è un’entità fisica. E’ infatti una struttura dati creata ed utilizzata dal programma applicativo per comunicare tra un processo client e un processo server (equivale a comunicare tra due socket create nei due socket create nei due lati di comunicazione)

![[Pasted image 20250323170213.png]]

Un socket address è composto da indirizzo IP e numero di porta
![[Pasted image 20250323170357.png|600]]

---
## Numeri di porta
I numeri di porta sono contenuti in $16 \text{ bit}$ ($0-65535$). Svariate porte sono usate da server noti (FTP 20, TELNET 23, SMTP 25, HTTP 80, POP3 110 etc.)

L’assegnamento delle porte segue queste regole:
- $0$ → non usata
- $1-255$ → riservate per processi noti
- $256-1023$ → riservate per altri processi
- $1024-65535$ → dedicate alle app utente

---
## Individuare i socket address
L’interazione tra client e server è **bidirezionale**. E’ necessaria quindi una coppia di indirizzi socket: **locale** (mittente) e **remoto** (destinatario); l’indirizzo locale in una direzione e l’indirizzo remoto nell’altra

### Individuare i socket address lato client
Il client ha bisogno di un socket address locale (client) e uno remoto (server) per comunicare

Il socket address **locale** viene **fornito dal sistema operativo**, infatti il SO conosce l’indirzzo IP del computer su cui il client è in esecuzione e il numero di porta è assegnato temporaneamente dal sistema operativo (numero di porta effimero, non viene utilizzato da latri processi)

Per quanto riguarda il socket address **remoto** il numero di porta è noto in base all’applicazione, mentre l’indirizzo IP è fornito dal DNS (oppure porta e indirizzo noti al programmatore quando si vuole verificare il corretto funzionamento di un’applicazione)

### Individuare i socket address lato server
Il server ha bisogno di un socket address locale (client) e uno remoto (server) per comunicare

Il socket address **locale** viene fornito dal sistema operativo, infatti il SO conosce l’indirzzo IP del computer su cui il server è in esecuzione e il numero di porta è **assegnato dal progettista** (numero well known o scelto)

Il socket address remoto è il socket address locale del client che si connette e poiché numerosi client possono connettersi, il server non può conoscere a priori tutti i socket address, ma li trova all’interno del pacchetto di richiesta

>[!warning]
>Il socket address locale di un server non cambia (è fissato e rimane invariato), mentre il socket address remoto varia ad ogni interazione con client diversi (anche con stesso client su connessioni diverse). Infatti se mi connetto da due browser allo stesso server cambierà il socket (hanno porte diverse); quindi dal server si riceveranno due risposte su due porte diverse

---
## Servizi di trasporto
Esistono due tipi di servizi di trasporto:
- affidabile → TCP
- non affidabile → UDP

### Servizio di TCP
Il servizio TCP è **orientato alla connessione** (è richiesto un setup fra i processi client e server)