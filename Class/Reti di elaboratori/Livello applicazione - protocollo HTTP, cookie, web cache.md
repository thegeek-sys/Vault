---
Created: 2025-03-13
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Web e HTTP
Il **World Wide Web** (*WWW*) è un’applicazione Internet nata dalla necessità di scambio e condivisione di informazioni tra ricercatori universitari di varie nazioni.

Questa opera su richiesta (on demand, a differenza di proattivi in cui c’è sempre qualcosa in funzione) e permette di rendere informazioni disponibili facilmente. In particolare è caratterizzato dai collegamenti ipertestuali (*hyperlinks*) che tramite i motori di ricerca permettono di navigare su una grande quantità di siti

Componenti:
- Web client (es. browser) → interfaccia con l’utente
- Web server (es. Apache)
- HTML → linguaggio standard per pagine web
- HTTP → protocollo per la comunicazione tra client e server Web (indipendente dal linguaggio di programmazione)

### Architettura generale di un browser
![[Pasted image 20250313235432.png|600]]

### Terminologia
Una pagina web è costituita da **oggetti** (es. file HTML, immagine, applet Java, …).
Una pagina web è formata da un **file base HTML** (*HyperText Markup Language*) che include diversi oggetti referenziati
Ogni oggetto è referenziato da un **URL** (*Uniform Resource Locator*), ad esempio `www.someschool.edu/someDept/pic.gif`

#### URL
Un **Uniform Resource Locator** (URL) è composto da 3 parti:
1. Protocollo
2. Nome della macchina in cui è situata la pagina
3. Percorso del file (locale alla macchina)
E’ possibile inoltre specificare una **porta** diversa da quella standard

![[Pasted image 20250313235915.png|500]]

### Documenti Web
Esistono tre tipi di documenti web:
- documento statico → contenuto predeterminato memorizzato sul server
- documento dinamico → creato dal web server alla recezione della richiesta
- documento attivo → contiene script o programmi che verranno eseguiti nel browser (lato client)

---
## Panoramica su HTTP (RFC 2616)
L’**HTTP** (*hypertext trasfer protocol*) è un protocollo a livello di applicazione del Web che regola la comunicazione tra client e server. In questo caso infatti il **client** richiede gli oggetti del Web, mentre il **server** invia gli oggetti in risposta ad una richiesta

>[!info] HTTP definisce in che modo i client web richiedono pagine ai server web e come questi le trasferiscono ai client

![[Pasted image 20250314000430.png|300]]
Il server può servire più richiede provenienti anche da client diversi

Dal lato client si ha che:
1. il browser determina l’URL ed estrae host e filename
2. esegue connessione TCP alla porta 80 dell’host indicato nella URL
3. invia richiesta per il file
4. riceve il file dal server
5. chiude la connessione
6. visualizza il file

Dal lato server si ha che:
1. accetta una connessione TCP da un client
2. riceve il nome del file richiesto
3. recupera il file dal disco
4. invia il file al client
5. rilascia la connessione

---
## Connessioni HTTP
Le connessioni HTTP si differenziano in:
- **connessioni non persistenti** (vecchio) → per ogni oggetto trasmesso viene aperta una relativa connessione TCP (al termine dell’invio dell’oggetto la connessione viene chiusa)
- **connessioni persistenti** → in questo caso non è necessario aprire una connessione per ogni oggetto, ma mi è sufficiente aprire una sola connessione e inviare tutte le richieste necessarie (la connessione viene chiusa quando rimane inattiva per un lasso di tempo configurabile)

>[!example] Connessioni non persistenti
>Supponiamo che l’utente immette l’URL `www.someSchool.edu/someDepartment/home.index` che contiene testo e riferimenti a 10 immagini jpeg
>
>1. il *client*  HTTP inizializza una connessione TCP con il processo server HTTP a `www.someSchool.edu` sulla porta 80
>2. il *server* HTTP all’host `www.someSchool.edu` in attesa di una connessione TCP alla porta 80 “accetta” la connessione e avvisa il client
>3. il *client* HTTP trasmette un messaggio di richiesta (con l’URL) nella socket della connessione TCP. Il messaggio indica che il client vuole l’oggetto `someDepartment/home.index`
>4. il *server* HTTP riceve il messaggio di richiesta, crea il messaggio di risposta che contiene l’oggetto richiesto e invia il messaggio nella sua socket
>5. il *server* HTTP chiude la connessione TCP
>6. il *client* HTTP riceve il messaggio di risposta che contiene il file html e visualizza il documento html. Esamina il file html, trova i riferimenti a 10 oggetti jpeg
>7. i passi 1-5 sono ripetuti per ciascuno dei 10 oggetti jpeg

---
## Schema del tempo di risposta
Il **RTT** (*Round Trip Time*) è il tempo impiegato da un piccolo pacchetto per andare dal client al server e ritornare al client. Questo include i ritardi di propagazione, accodamento e elaborazione del pacchetto

Dunque il **tempo di risposta** è composto da:
- un RTT per inizializzare la connessione TCP
- un RTT per la richiesta HTTP e i primi byte della risposta HTTP
- tempo di trasmissione del file

![[Pasted image 20250314002121.png|400]]

### Connessioni persistenti
Come già detto il server lascia la connessione TCP aperta dopo l’invio di una risposta, questo ci permette di diminuire di gran lunga il numero di RTT, verrà infatti effettuata un solo RTT di connessione per tutti gli oggetti referenziali (+ un RTT per ogni oggetto ricevuto dal server)

Svantaggi delle connessioni non persistenti:
- richiedono 2 RTT per oggetto
- overhead del sistema operativo per ogni connessione TCP
- i browser spesso aprono connessioni TCP parallele per caricare gli oggetti referenziati (da 5 a 10 connessioni)

---
## Formato generale dei messaggi di richiesta HTTP
