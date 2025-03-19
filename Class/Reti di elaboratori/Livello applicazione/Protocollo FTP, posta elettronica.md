---
Created: 2025-03-18
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello applicazione]]"
Completed:
---
---
## Introduction
In questa sezione affronteremo i protocolli:
- FTP
- SMTP
- POP3
- IMAP

---
## File Transfer Protocol (FTP)
Il **File Trasfer Protocol** (*FTP*) è un protocollo di trasferimento file da/a un host remoto.

> [!example] Utilizzo
> Il comando per accedere ed essere autorizzato a scambiare informazioni con l’host remoto è:
> ```bash
> ftp NomeHost
> # vengono richiesti nome utente e password
> ```
> 
> Trasferimento di un file da un host remoto:
> ```bash
> ftp> get file1.txt
> ```
> 
> Trasferimento di un file a un host remoto:
> ```bash
> ftp> put file3.txt
> ```

![[Pasted image 20250318212416.png|center|500]]

Il modello è basato su client/server:
- **client** → lato che inizia il trasferimento (a/da un host remoto)
- **server** → host remoto

Quando l’utente fornisce il nome dell’host remoto (`ftp NomeHost`), il processo client FTP stabilisce una connessione TCP sulla porta 21 con il processo server FTP.
Stabilita la connessione, il client fornisce nome utente e password che vengon inviate sulla connessione TCP come parte dei comandi
Ottenuta l’autorizzazione del server il client può inviare uno o più file memorizzati nel filesystem locale verso quello remoto (o viceversa)

---
## FTP client e server
![[Pasted image 20250318212817.png|center|500]]
Nel protocollo FTP si distinguono due tipi di connessione:
- **connessione di controllo** → si occupa delle informazioni di controllo del trasferimento e usa regole molto semplici, così che lo scambio di informazioni si riduce allo scambio di  una riga di comando (o risposta) per ogni interazione
- **connessione dati** → si occupa del trasferimento file

![[Pasted image 20250318213621.png|center|350]]
### Connessione di controllo
La **connessione di controllo** (porta 21) viene usata per inviare informazioni di controllo (vengono ad esempio trasferite identificativo utente, password, comandi per cambiare cartella, comandi per richiedere trasferimento di file).
Tutti i comandi eseguiti dall’utente sono trasferiti sulla connessione di controllo

Questa connessione viene definita *out of band* poiché è separata dalla connessione per il trasferimento di file. Invece HTTP usa la stessa connessione per messaggi di richiesta e risposta e file, per cui si dice che invia la informazioni di controllo *in band*

Il server FTP inoltre mantiene lo stato riguardo directory corrente e autenticazione precedente

### Connessione dati
Quando il server riceve un comando per trasferire un file, apre una connessione dati TCP sulla porta 20 con il client. Dopo il trasferimento di un file, il server chiude la connessione (si crea una nuova connessione per ogni file trasferito all’interno della sessione).
La connessione dati viene aperta dal server e utilizzata per il vero e proprio invio del file.

---
## Comandi e risposte FTP
Esiste una corrispondenza uno a uno tra il comando immesso dall’utente e quello FTP inviato sulla connessione di controllo. Ciascun domando è seguito da una risposta spedita dal server al client (codice di ritorno)

| Comandi comuni                                                                                                            | Codici di ritorno comuni                              |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Inviati come testo ASCII sulla connessione di controllo                                                                   | Codice di stato ed espressione (come in HTTP)         |
| `USER username`                                                                                                           | `331 Username OK, password required`                  |
| `PASS password`                                                                                                           |                                                       |
| `LIST`<br>elenca i file della directory corrente, la lista di file viene inviata dal server su una nuova connessione dati | `125 data connection already open; transfer starting` |
| `RETR filename`<br>recupera (`get`) un file dalla directory corrente                                                      | `425 Can't open data connection`                      |
| `STOR filename`<br>memorizza (`put`) un file nell’host remoto                                                             | `452 Error writing file`                              |
### Principali comandi FTP
La codifica standard per i comandi e le risposte FTP è la NVT ASCII
![[Pasted image 20250319091403.png]]

### Esempi di risposte FTP
Le risposte sono composte da due parti: un **numero di 3 cifre e un testo**.
La parte numerica costituisce il codice della risposta, quella di testo contiene i parametri necessari o informazioni supplementari
La tabella riporta alcuni codici (non il testo)
![[Pasted image 20250319091555.png]]

### Esempio
La connessione dati viene aperta e chiusa per ogni trasferimento di file (porte 1267-20)
![[Pasted image 20250319091713.png]]

---
## Posta elettronica: scenario classico
Ci sono tre componenti principali:
- **User agent** → usato per scrivere e inviare un messaggio o leggerlo
- **Message Transfer Agent** → usato per trasferire il messaggio attraverso Internet
- **Message Access Agent** → usato per leggere la mail in arrivo

![[Pasted image 20250319100251.png]]

### User agent
Lo **user agent** (detto anche “mail reader”) viene attivato dall’utente o da un timer: se c’è una nuova mail informa l’utente; è inoltre responsabile di composizione, editing e lettura dei messaggi di posta elettronica

I messaggi in uscita vengono quindi memorizzati sul server attraverso il **Mail Transfer Agent**
Sia i messaggi in uscita che in arrivo però sono memorizzati sul server

### Mail Transfer Agent
Il **Mail Transfer Agent** dunque è il software lato **server** che gestisce l’invio e la ricezione della mail. Questo è composto da una **casella di posta** (*mailbox*) che contiene i messaggi in arrivo per l’utente e da una **coda dei messaggi** da trasmettere (con tentativi ogni $x$ minuti per alcuni giorni)

![[Pasted image 20250319101234.png|center|350]]

Per comunicare tra server di posta il MTA utilizza il protocollo **SMTP** (*Simple Mail Transfer Protocol*)

---
## SMTP (RFC 5321)
Il protocollo **SMTP** usa TCP per trasferire in modo affidabile i messaggi di posta elettronica dal client al server, e lo fa attraverso la porta 25
Si tratta di un trasferimento diretto: dal server trasmittente al server ricevente

Ogni trasferimento di questo tipo è caratterizzato da tre fasi:
1. **Handshaking**
2. **trasferimento dei messaggi**
3. **chiusura**

Il formato del comando/risposta:
- comandi → testo ASCII
- risposta → codice di stato ed espressione

Inoltre i messaggi devono essere nel formato ASCII

>[!example] Scenario: Alice invia un messaggio a Roberto
>1. Alice usa il suo user agent per comporre il messaggio da inviare “a” `rob@someshool.edu`
>2. Lo user agent di Alice invia un messaggio al server di posta di Alice; il messaggio è posto nella coda dei messaggi
>3. Il lato client di SMTP apre una connessione TCP con il server di risposta di Roberto
>4. Il client SMTP invia il messaggio di alice sulla connessione TCP
>5. Il server i posta di Roberto riceve il messaggio e lo pone nella casella di posta di Roberto
>6. Roberto invoca il suo agente utente per leggere il messaggio
>
>![[Pasted image 20250319102220.png]]
>
>**Protocolli utilizzati**:
>![[Pasted image 20250319102252.png]]

### Scambio di messaggi a livello di protocollo
Il client SMTP (che gira sull’host server di posta in invio) fa stabilire una connessione sulla porta 25 verso il server SMTP (che gira sull’host server di posta in ricezione)
Se il server è **inattivo** il client riprova più tardi, mentre se il server è **attivo** viene stabilita la connessione

Il server e il client effettuano una forma di handshaking (il client indica indirizzo email del mittente e del destinatario). Quindi il client invia il messaggio e il messaggio arriva al server destinatario grazie all’affidabilità del TCP
Se ci sono altri messaggi si usa la stessa connessione (connessione persistente), altrimenti il client invia richiesta di chiusura connessione

>[!example] Esempio di interazione SMTP
>- Client → `crepes.fr`
>- Server → `hamburger.edu`
>
>La seguente transazione inizia appena si stabilisce la connessione TCP
>![[Pasted image 20250319102941.png]]
>
>>[!info] Note
>>- SMTP usa connessioni persistenti (ripete i passi da `MAIL FROM:`)
>>- SMTP richiede che il messaggio (intestazione e corpo) sia nel formato ASCII a 7 bit
>>- Il server SMTP usa **`CRLF.CRLF`** per determinare la fine del messaggio

Confronto con HTTP:

| HTTP                                                                     | SMTP                                                                           |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| utilizzato per trasferire file da un host all’altro                      | utilizzato per trasferire file da un host all’altro                            |
| `pull`<br>gli utenti scaricano i file e inizializzano le connessioni TCP | `push`<br>il server di posta spedisce il file e inizializza la connessione TCP |
| ciascun oggetto è incapsulato nel suo messaggio di risposta              | più oggetti vengono trasmessi in un unico messaggio                            |

---
## Formato dei messaggi di posta elettronica
![[Pasted image 20250319104133.png]]

>[!warning] Differenti dai comandi SMTP

>[!example] Esempio: fasi trasferimento
>![[Pasted image 20250319104226.png]]

---
## Protocollo MIME
Come si possono inviare messaggi in formati non ASCII? Bisogna convertire i dati

![[Pasted image 20250319105054.png]]

Per inviare contenuti diversi dal testo ASCII si usano interazioni aggiuntive
Il **MIME** (*Multipurpose Internet Mail Extension*) sono estensioni di messaggi di posta multimediali

### Formato del messaggio inviato
Nell’intestazione dei messaggi (in caso di contenuti MIME) viene dichiarato il tipo di contenuto MIME

![[Pasted image 20250319105319.png]]

### Formato del messaggio ricevuto
Un’altra classe di 