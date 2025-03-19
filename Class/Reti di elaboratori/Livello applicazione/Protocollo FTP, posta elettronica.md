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

### FTP client e server
![[Pasted image 20250318212817.png|center|500]]
Nel protocollo FTP si distinguono due tipi di connessione:
- **connessione di controllo** → si occupa delle informazioni di controllo del trasferimento e usa regole molto semplici, così che lo scambio di informazioni si riduce allo scambio di  una riga di comando (o risposta) per ogni interazione
- **connessione dati** → si occupa del trasferimento file

![[Pasted image 20250318213621.png|center|350]]
#### Connessione di controllo
La **connessione di controllo** (porta 21) viene usata per inviare informazioni di controllo (vengono ad esempio trasferite identificativo utente, password, comandi per cambiare cartella, comandi per richiedere trasferimento di file).
Tutti i comandi eseguiti dall’utente sono trasferiti sulla connessione di controllo

Questa connessione viene definita *out of band* poiché è separata dalla connessione per il trasferimento di file. Invece HTTP usa la stessa connessione per messaggi di richiesta e risposta e file, per cui si dice che invia la informazioni di controllo *in band*

Il server FTP inoltre mantiene lo stato riguardo directory corrente e autenticazione precedente

#### Connessione dati
Quando il server riceve un comando per trasferire un file, apre una connessione dati TCP sulla porta 20 con il client. Dopo il trasferimento di un file, il server chiude la connessione (si crea una nuova connessione per ogni file trasferito all’interno della sessione).
La connessione dati viene aperta dal server e utilizzata per il vero e proprio invio del file.

### Comandi e risposte FTP
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
