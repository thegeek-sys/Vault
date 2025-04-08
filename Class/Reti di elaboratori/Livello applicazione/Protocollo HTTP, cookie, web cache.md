---
Created: 2025-03-13
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Index
- [[#Web e HTTP|Web e HTTP]]
	- [[#Web e HTTP#Architettura generale di un browser|Architettura generale di un browser]]
	- [[#Web e HTTP#Terminologia|Terminologia]]
		- [[#Terminologia#URL|URL]]
	- [[#Web e HTTP#Documenti Web|Documenti Web]]
- [[#Panoramica su HTTP (RFC 2616)|Panoramica su HTTP (RFC 2616)]]
- [[#Connessioni HTTP|Connessioni HTTP]]
- [[#Schema del tempo di risposta|Schema del tempo di risposta]]
	- [[#Schema del tempo di risposta#Connessioni persistenti|Connessioni persistenti]]
- [[#Formato generale dei messaggi di richiesta HTTP|Formato generale dei messaggi di richiesta HTTP]]
- [[#Upload dell’input di un form|Upload dell’input di un form]]
	- [[#Upload dell’input di un form#Tipi di metodi (HTTP/1.1)|Tipi di metodi (HTTP/1.1)]]
	- [[#Upload dell’input di un form#Intestazioni nella richiesta|Intestazioni nella richiesta]]
- [[#Formato generale dei messaggi di risposta HTTP|Formato generale dei messaggi di risposta HTTP]]
	- [[#Formato generale dei messaggi di risposta HTTP#Codici di stato della risposta HTTP|Codici di stato della risposta HTTP]]
	- [[#Formato generale dei messaggi di risposta HTTP#Intestazioni nella risposta|Intestazioni nella risposta]]
- [[#Esempio GET|Esempio GET]]
- [[#Esempio PUT|Esempio PUT]]
- [[#Cookie|Cookie]]
	- [[#Cookie#Sessione|Sessione]]
	- [[#Cookie#Interazione utente-server|Interazione utente-server]]
	- [[#Cookie#Nel dettaglio|Nel dettaglio]]
	- [[#Cookie#Durata di un cookie|Durata di un cookie]]
	- [[#Cookie#Altre soluzione per mantenere lo stato|Altre soluzione per mantenere lo stato]]
- [[#Web caching|Web caching]]
	- [[#Web caching#Browser caching|Browser caching]]
	- [[#Web caching#Server proxy|Server proxy]]
		- [[#Server proxy#Server proxy in una LAN|Server proxy in una LAN]]
- [[#Esempio in assenza di cache|Esempio in assenza di cache]]
	- [[#Esempio in assenza di cache#Ipotesi|Ipotesi]]
	- [[#Esempio in assenza di cache#Stima del tempo di risposta|Stima del tempo di risposta]]
	- [[#Esempio in assenza di cache#Soluzione possibile|Soluzione possibile]]
		- [[#Soluzione possibile#Conseguenze|Conseguenze]]
- [[#Esempio in presenza di cache|Esempio in presenza di cache]]
- [[#Inserimento di un oggetto in cache|Inserimento di un oggetto in cache]]
- [[#Validazione dell’oggetto|Validazione dell’oggetto]]
	- [[#Validazione dell’oggetto#GET condizionale|GET condizionale]]
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
Il server può servire più richieste provenienti anche da client diversi

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
![[Pasted image 20250314002821.png|600]]

>[!example] Esempio richiesta HTTP
>Messaggio di richiesta HTTP inviato dal client
>![[Pasted image 20250314003116.png|500]]
>![[Pasted image 20250314003155.png|550]]

---
## Upload dell’input di un form
Per inviare un input al server viene usato spesso il metodo **POST**, tramite il quale l’input arriva al server nel corpo dell’entità.
In alternativa si usa il metodo **GET** inserendo nel campo URL l’input richiesto `www.somesite.com/animalsearch?monkeys&banana`

### Tipi di metodi (HTTP/1.1)
![[Pasted image 20250314003514.png|500]]

### Intestazioni nella richiesta
![[Pasted image 20250314003550.png]]

---
## Formato generale dei messaggi di risposta HTTP
![[Pasted image 20250314003639.png|600]]

>[!warning] Ricordare come sono strutturate richieste/risposte (per esame)

>[!example] Messaggio di risposta HTTP
>![[Pasted image 20250314003726.png|500]]
>![[Pasted image 20250314003756.png|550]]

### Codici di stato della risposta HTTP
Nella prima riga del messaggio di risposta server→client si trovano i **codici di stato**. Sotto riportati alcuni codici di stato e le relative e le relative espressioni:
- `200 OK` → la richiesta ha avuto successo; l’oggetto richiesto viene inviato nella risposta
- `301 Moved Permanently` → l’oggetto richiesto è stato trasferito; la nuova posizione è specificata nell’intestazione `Location`: della risposta
- `400 Bad Request` → il messaggio di richiesta non è stato compreso dal server
- `404 Not Found` → il documento richiesto non si trova su questo server
- `505 HTTP Version Not Supported` → il server non ha la versione di protocollo HTTP

| Codice | Significato      | Esempi                                                      |
| ------ | ---------------- | ----------------------------------------------------------- |
| `1xx`  | Informazione     | 100 = server accetta di gestire la richiesta client         |
| `2xx`  | Successo         | 200 = richiesta con successo; 204 = contenuto non presente  |
| `3xx`  | Reindirizzamento | 301 = pagina spostata; 304 = pagina cachata ancora valida   |
| `4xx`  | Errore client    | 403 = pagina proibita; 404 = pagina non trovata             |
| `5xx`  | Errore server    | 500 = errore interno server; 503 = prova di nuovo più tardi |

### Intestazioni nella risposta
![[Pasted image 20250314004344.png]]

---
## Esempio GET
Il client preleva un documento: viene usato il metodo GET per ottenere l’immagine individuata dal percorso `/usr/bin/image1`.

![[Pasted image 20250314004608.png|500]]

La riga di richiesta contiene il metodo (GET), l’URL e la versione (1.1) del protocollo HTTP. L’intestazione è costituita da due righe in cui si specifica che il client accetta immagini nei formati GIF e JPEG. Il messaggio di richiesta non ha corpo.
Il messaggio di risposta contiene la riga di stato e quattro righe di intestazione che contengono la data, il server, il metodo di codifica del contenuto (la versione MIME, argomento che verrà descritto nel paragrafo dedicato alla posta elettronica) e la lunghezza del documento. Il corpo del messaggio segue l’intestazione.

---
## Esempio PUT
Il client spedisce al server una pagina Web da pubblicare. Si utilizza il metodo PUT.

![[Pasted image 20250314004732.png|500]]

La riga di richiesta contiene il metodo (PUT), l’URL e la versione (1.1) del protocollo HTTP. L’intestazione è costituita da quattro righe d’intestazione. Il corpo del messaggio di richiesta contiene la pagina Web inviata.
Il messaggio di risposta contiene la riga di stato e quattro righe di intestazione. Il documento creato, un documento CGI, è incluso nel corpo del messaggio di risposta.

---
## Cookie
L’HTTP è un protocollo “senza stato” (*stateless*), infatti il server, una volta servito il client, se ne dimentica e non mantiene informazioni sulle richieste fatte
Però i protocolli che mantengono lo stato sono complessi, poiché la storia passata deve essere memorizzata e inoltre in caso di errore (client e/o server) durante una richiesta, si ha un disallineamento tra quello client e server

Nonostante tutto ciò ci sono molti casi in cui il server ha bisogno di **ricordarsi degli utenti**. Una prima soluzione potrebbe essere l’utilizzo degli indirizzi IP, ma non risultano adatti poiché con l’introduzione del *NAT*, da una stessa rete LAN esce un solo indirizzo IP per tutti i client

La soluzione è quindi il **Cookie** (RFC 6265).
Il meccanismo dei Cookie rappresenta un modo per creare una sessione di richieste e risposte HTTP “con stato” (*stateful*)
Cosa possono contenere i cookie:
- autorizzazione
- carta per acquisti
- preferenze dell’utente
- stato della sessione dell’utente (e-mail)

### Sessione
Ci possono essere diversi tipi di sessione in base al tipo di informazioni scambiate e la natura del sito.
Caratteristiche generali di una sessione:
- ogni sessione ha un inizio e una fine
- ogni sessione ha un tempo di vita relativamente corto
- sia il client che il server possono chiudere la sessione
- la sessione è implicita nello scambio di informazioni di stato

>[!info] Sessione vs. connessione
>Per "sessione" NON si intende connessione persistente, ma una sessione logica creata da richieste e risposte HTTP. Una sessione può essere creata su connessioni persistenti e non persistenti

### Interazione utente-server
I passaggi dell’interazione utente-server sono:
1. una riga di intestazione nel messaggio di risposta HTTP
2. una riga di intestazione nel messaggio di richiesta HTTP
3. un file cookie mantenuto sul sistema terminale dell’utente e gestito dal browser dell’utente
4. un database sul server

>[!example] Esempio
>- L’utente A accede sempre a Internet dallo stesso PC (non necessariamente con lo stesso IP)
>- Visita per la prima volta un particolare sito di commercio elettronico
>- Quando la richiesta HTTP iniziale giunge al sito, il sito crea un identificativo unico (ID) e una entry nel database per ID
>- L’utente A invierà ogni futura richiesta inserendo l’ID nella richiesta

>[!example] Cookie in un negozio online
>![[Pasted image 20250314010144.png|500]]

### Nel dettaglio
Il server mantiene tutte le informazioni riguardanti il client su un file e gli assegna un identificatore (cookie) che viene fornito al client. Il cookie inviato al client è un **identificatore di sessione** (*SID*)

Per evitare che il cookie sia utilizzato da utenti “maligni” l’identificatore è composto da una stringa di numeri

>[!example]
>```
>== Server -> User Agent ==
>Set-Cookie: SID=31d4d96e407aad42
>```

Il client ogni volta che manda una richiesta al server fornisce il suo identificatore (cookie): il browser consulta il file cookie, estrae il numero di cookie per il sito che si vuole visitare e lo inserisce nella richiesta HTTP

>[!example]
>```
>== User Agent -> Server ==
>Cookie: SID=31d4d96e407aad42
>```

Il server mediante il cookie fornito dal client accede al relativo file e fornisce risposte personalizzate

![[Pasted image 20250314010652.png]]

### Durata di un cookie
Il server chiude una sessione inviando al client una intestazione `Set-Cookie` nel messaggio con `Max-Age=0`. Infatti l’attributo `Max-Age` definisce il tempo di vita in secondi di un cookie. Dopo delta secondi il client dovrebbe rimuovere il cookie. Il valore zero indica che il cookie deve essere rimosso subito.

### Altre soluzione per mantenere lo stato
Per mantenere lo stato (e quindi creare una sessione) il client mantiene tutte le informazioni sullo stato della sessione e le inserisce in ogni richiesta inviata al server tramite metodo POST o inserendole nella URL

Vantaggi:
- facile da implementare
- non richiede l’introduzione di particolare funzionalità sul server

Svantaggi:
- può generare lo scambio di grandi quantità di dati
- le risorse del server devono essere re-inizializzate ad ogni richiesta

---
## Web caching
L’obiettivo è quello di **migliorare le prestazioni dell’applicazione web**.
Un modo semplice consiste nel salvare  le pagine richieste per riutilizzarle in seguito senza doverle richiedere al server (efficiente con pagine che vengono visitate molto spesso)

L’accumulo delle pagine per un utilizzo successivo è definito **caching**
Il caching può essere eseguito da:
- **browser**
- **proxy**

>[!question] Perché il caching web?
>- Riduce i tempi di risposta alle richieste dei client
>- Riduce il traffico sul collegamento di accesso a Internet
>- Internet arricchita di cache consente ai provider meno efficienti di fornire dati con efficacia

### Browser caching
Il browser può mantenere una cache personalizzabile delle pagine visitate

Esistono vari meccanismi per la gestione della cache locale:
- L’utente può impostare il n**umero di giorni dopo i quali i contenuti della cache vengono cancellati** e l’eventuale gestione
- La pagina può essere **mantenuta in cache in base alla sua ultima modifica** (es. modificata un’ora prima → mantenuta per un’ora, un giorno, etc.)
- Si possono utilizzare informazioni nei **campi intestazione dei messaggi** per gestire la cache (es. campo `Expires` specifica la scadenza dopo la quale la pagina è considerata obsoleta, ma non sempre è rispettato dai browser)

### Server proxy
Il server proxy ha una memoria per mantenere copie della pagine visitate
Il browser può essere configurato per inviare le richieste dell’utente alla cache, nel caso in cui fosse presente, la cache fornisce l’oggetto, altrimenti la cache richiede l’oggetto al server d’origine e poi lo inoltra al client

![[Pasted image 20250314012122.png|400]]

#### Server proxy in una LAN
![[Pasted image 20250314012155.png|550]]
Anche gli ISP possono mantenere un proxy per le richieste dei vari utenti

---
## Esempio in assenza di cache
![[Pasted image 20250314100050.png|center|250]]
### Ipotesi
Dimensione media dell’oggetto: $1\text{ Mb}$
Frequenza media di richieste dai browser istituzionali ai server d’origine: $15 \text{ richieste al secondo}$
Ritardo per recuperare un oggetto sulla rete internet (*Internet delay*): $2\text{ sec}$
Tempo totale di risposta (*total delay*): $\text{LAN delay}+\text{access delay}+\text{Internet delay}$

### Stima del tempo di risposta
Valutiamo l’intensità di traffico $\frac{La}{R}$
$$
\text{utilizzo sulla LAN}=\frac{15\text{ req/s}\cdot 1\text{ Mb/req}}{100\text{ Mbps}}=15\%
$$
$$
\text{utilizzo sul collegamento d'accesso}=\frac{15\text{ req/s}\cdot 1\text{ Mb}}{15 \text{ Mbps}}=100\%
$$
Ricordiamo che quando l’intensità di traffico riempie il collegamento, il ritardo tende ad aumentare, quindi si ha:
$$
\begin{align}
\text{ritardo totale}&=\text{ritardo di Internet}+\text{ritardo di accesso}+\text{ritardo della LAN} \\
&=2\text{ sec}+\text{minuti}+\text{millisecondi}
\end{align}
$$
### Soluzione possibile
![[Pasted image 20250314100029.png|250]]
Una possibile soluzione sta nell’aumentare l’ampiezza di banda del collegamento d’accesso a $100\text{ Mbps}$ per esempio
#### Conseguenze
$$
\text{utilizzo sulla LAN}=15\%
$$
$$
\text{utilizzo sul collegamento d'accesso}=15\%
$$
$$

\begin{align}
\text{ritardo totale}&=\text{ritardo di Internet}+\text{ritardo di accesso}+\text{ritardo della LAN} \\
&=2\text{ sec}+\text{millisecondi}+\text{millisecondi}
\end{align}
$$
Ma non sempre è attuabile e comunque risulta costoso aggiornare il collegamento

---
## Esempio in presenza di cache
Una soluzione alternativa all’esempio precedente consiste nell’installare la cache
![[Pasted image 20250314100541.png|250]]
Supponiamo una percentuale di successo (*hit rate*) pari a $0,4$.
Come conseguenza si ha che:
- il $40\%$ delle richieste sarà soddisfatto quasi immediatamente (circa $10\text{ ms}$)
- il $60\%$ delle richieste sarà soddisfatto dal server d’origine
- l’utilizzo del collegamento d’accesso si è ridotto del $60\%$, determinando ritardi trascurabili (circa $10\text{ ms}$)
$$
\begin{align}
\text{ritardo totale medio}&=\text{ritardo di Internet}+\text{ritardo di accesso}+\text{ritardo della LAN} \\
&=0,6\cdot 2,01\text{ sec}+0,4\cdot 0,01\text{ sec} \sim 1,2\text{ sec}
\end{align}
$$

---
## Inserimento di un oggetto in cache
Il **client** invia un messaggio di richiesta HTTP alla cache
```
GET /page/figure.gif
Host: www.sito.com
```
La cache non ha l’oggetto, quindi invia la richiesta HTTP al **server**. Quindi il server invia la risposta  HTTP alla cache
```
HTTP/1.1 200 OK
Date: ...
...
Last-Modified: Wed, 2 Jul 2008 9:23:24
```
La cache così memorizza la pagina per richieste future, mantenendo la data di ultima modifica. Infine la cache invia la risposta al client

---
## Validazione dell’oggetto
Il **client** invia un messaggio di richiesta HTTP alla cache
```
GET /page/figure.gif
Host: www.sito.com
```
La cache ha l’oggetto, a prima di inviarlo al client deve verificare che non sia scaduto (modificato sul server di origine).
Quindi la cache esegue una richiesta verso il Web server che mantiene l’oggetto per verificarne la validità mediante un **GET condizionale**
### GET condizionale
Il GET condizionale permette di inviare una richiesta se e solo se determinati requisiti sono soddisfatti
![[Pasted image 20250314101320.png|250]]
In questo caso la cache specifica la data della copia dell’oggetto nella richiesta HTTP
```
If-modified-since
```
Ora la risposta del server non contiene l’oggetto se la copia della cache è aggiornata
```
HTTP/1.1 304 Not Modified
```
