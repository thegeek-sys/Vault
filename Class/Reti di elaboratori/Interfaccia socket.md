---
Created: 2025-04-04
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Nel paradigma client/server la comunicazione a livello applicazione avviene tra due programmi applicativi in esecuzione chiamati processi: un client e un server
- un client è un programma in esecuzione che inizia la comunicazione inviando una richiesta
- un server è un altro programma applicativo che attende le richieste dai client

---
## API
Un linguaggio di programmazione prevede un insieme di istruzioni matematiche (un insieme di istruzioni per la manipolazione delle stringhe etc.)
Se si vuole sviluppare un programma capace di comunicare con un altro programma, è necessario un nuovi insieme di istruzioni per chiedere ai primi quattro livelli dello stack TCP/IP di aprire la connessione, inviare/ricevere dati e chiudere la connessione

Un insieme di istruzioni di questo tipo viene chiamato **API** (*Application Programming Interface*)

![[Pasted image 20250404115437.png|550]]

---
## Comunicazione tra processi
La comunicazione tra i processi avviene tramite il **socket**

Il **socket** appare come un terminale o un file ma non è un’entità fisica. E’ infatti una struttura dati creata ed utilizzata dal programma applicativo per comunicare tra un processo client e un processo server (equivale a comunicare tra due socket create nei due socket create nei due lati di comunicazione)

![[Pasted image 20250323170213.png]]

Un socket address è composto da indirizzo IP e numero di porta
![[Pasted image 20250323170357.png|500]]

---
## Indirizzamento dei processi
Affinché un processo su un host invii un messaggio a un processo su un altro host, il mittente deve identificare il processo destinatario
Un host ha un indirizzo IP univoco a $32\text{ bit}$ ma non è sufficiente questo per identificare anche il processo (sullo stesso host possono essere in esecuzione più processi) ma è necessario anche il **numero di porta** associato al processo

### Come viene recapitato un pacchetto all’applicazione
![[Pasted image 20250404120414.png|400]]

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
## Utilizzo dei servizi di livello trasporto
Una coppia di processi fornisce servizi agli utenti Internet, siano questi persone o applicazioni.
La coppia di processi, tuttavia, deve utilizzare i servizi offerti dal livello trasporto per la comunicazione, poiché non vi è comunicazione fisica a livello applicazione

Nel livello trasporto della pila di protocolli TCP/IP sono previsti due protocolli principali:
- protocollo UDP
- protocollo TCP

### Quale servizio richiede l’applicazione?
#### Perdita di dati
Alcune applicazione (es. audio) possono tollerare qualche perdita, mentre altre applicazioni (es. trasferimento dati) richiedono un trasferimento dati affidabile al 100%
#### Temporizzazione
Alcune applicazioni (es. giochi, Internet) per essere “realistiche” richiedono piccoli ritardi, mentre altre applicazioni (es. posta elettronica) non hanno particolari requisiti di temporizzazione
#### Throughput
Alcune applicazioni (es. multimediali) per essere efficaci richiedono un’ampiezza di banda minima, mentre altre applicazioni (“le applicazioni elastiche”) utilizzano l’ampiezza di banda che si rende disponibile
#### Sicurezza
Cifratura, integrità dei dati, …

---
## Programmazione con socket
La **socket API** è stato introdotta in BDS4.1 UNIX nel 1981. Questa viene esplicitamente creata, usata, distribuita dalle applicazioni secondo il paradigma client/server

Si hanno due tipi di servizio di trasporto tramite socket API:
- datagramma inaffidabile
- affidabile, orientata ai byte

![[Pasted image 20250404122435.png|550]]

**Pre-requisiti per contattare il server**:
- il processo server deve essere in esecuzione
- il server deve aver creato un socket che dà il benvenuto al contatto con il client

**Il client contatta il server**:
- creando un socket TCP
- specificando l’indirizzo IP, il numero di porta del processo server
- una volta fatto ciò il client TCP stabilisce una connessione con il server TCP

![[Pasted image 20250404122919.png|400]]

Quando viene contattato dal client, il server TCP crea un nuovo socket per il processo server per comunicare con il client che consente al server id comunicare 