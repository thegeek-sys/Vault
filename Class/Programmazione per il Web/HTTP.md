---
Created: 2025-10-08
Related:
Class: "[[Programmazione per il Web]]"
---
---
## HyperText Transfer Protocol
L’**HTTP** è un protocollo di livello applicazione nello stack di protocolli Internet, la cui variante sicura è chiamata **HTTPS**

| Versione | Anno di introduzione | Stato attuale |
| -------- | -------------------- | ------------- |
| HTTP/0.9 | 1991                 | Obsoleto      |
| HTTP/1.0 | 1996                 | Obsoleto      |
| HTTP/1.1 | 1997                 | Standard      |
| HTTP/2   | 2015                 | Standard      |
| HTTP/3   | 2022                 | Standard      |

>[!example] Esempio richiesta
>- linea di richiesta → metodo HTTP, URI, versione protocollo
>- campi di intestazione della richiesta
>- corpo del messaggio (opzionale)
>
>```
>GET /hello.txt HTTP/1.1
>User-Agent: curl/7.64.1
>Host: www.example.com
>Accept-Language: en, it
>```

>[!example] Esempio risposta
>- stato di completamento (riguardo la richiesta)
>- contenuto (opzionale)
>
>```
>HTTP/1.1 200 OK
>Date: Mon, 27 Jul 2009 12:28:53 GMT
>Server: Apache
>Last-Modified: Wed, 22 Jul 2009 19:15:56 GMT
>ETag: "34aa387-d-1568eb00"
>Accept-Ranges: bytes
>Content-Length: 51
>Vary: Accept-Encoding
>Content-Type: text/plain
>Hello World! My content includes a trailing CRLF.
>```

### Client
Un **user agent** (UA) è un qualsiasi programma client che avvia una richiesta
### Server
L’**origin server** (O) è un programma che può originare risposte autorevoli per una data risorsa
### Intermediari
Di solito tra client e server sono presenti degli intermediari come ad esempio proxy, gateway o tunnel
### Cache
La cache corrisponde ad un archivio di messaggi di risposta precedenti. Per poterla usare bisogna dichiarare una risposta come **cachable**. Inoltre i proxy possono memorizzare le risposte nella cache mentre i tunnel no

---
## Metodi HTTP
- GET
- HEAD
- POST
- PUT
- DELETE
- CONNECT
- OPTIONS
- TRACE
- PATCH

### Proprietà
#### Safe
Un metodo che non ha effetti collaterali sulla risorsa è definito safe. Può tuttavia cambiare lo stato del server in altri modi (es. log)

Lo sono: GET, HEAD, OPTIONS e TRACE
#### Idempotent
Se richieste identiche multiple con quel metodo hanno lo stesso effetto di una singola richiesta sono definite idempotenti

Lo sono: PUT e DELETE
#### Cachable
Ovvero i metodi che possono consentire ad una cache di archiviare e utilizzare una risposta

Lo sono: GET, HEAD e POST

### PUT
Crea una nuova risorsa specificandola nella richiesta, o la sostituisce se l’URI esiste

```
PUT /course-descriptions/web-and-software-architecture
```

### GET
Richiede una rappresentazione dello stato di una risorsa

```
GET /course-descriptions/web-and-software-architecture
```

### POST
Crea o modifica un subordinato della risorsa indicata nell’URI. A differenza del PUT non sostituirà una risorsa esistente, ma ne crea una seconda

```
POST /announcements/
POST /announcements/{id}/comments/
POST /users/{id}/email
```

>[!warning]
>L’azione non deve necessariamente creare una nuova risorsa

### DELETE
Richieste che il server di origine rimuova l’associazione tra la risorsa target e la sua funzionalità attuale

```
DELETE /courses/web-and-software-architecture
```

### Altri metodi

| Metodo  | Descrizione                                                                         |
| ------- | ----------------------------------------------------------------------------------- |
| HEAD    | come GET, ma non trasferisce il contenuto della risposta                            |
| CONNECT | stabilisce un tunnel verso il server identificato dalla risorsa target              |
| OPTIONS | descrive le opzioni di comunicazione per la risorsa target                          |
| TRACE   | esegue un test di loop-back del messaggio lungo il percorso verso la risorsa target |
### Codici di stato della risposta
Il codice di stato descrive il risultato della richiesta e la semantica della risposta, permettendo di sapere se la richiesta ha avuto successo e quale contenuto è allegato (se presente)
#### 2xx successo
- `200 OK` → in una richiesta GET, la risposta conterrà un'entità corrispondente alla risorsa richiesta; in una richiesta POST, la risposta conterrà un'entità che descrive o contiene il risultato dell'azione
- `201 Created` → la richiesta è stata soddisfatta, risultando nella creazione di una nuova risorsa
- `204 No Content` → il server ha elaborato con successo la richiesta e non sta restituendo alcun contenuto
#### 3xx reindirizzamento
- `301 Moved Permanently` → questa e tutte le richieste future dovrebbero essere indirizzate all’URI fornito
- `302 Found` → guarda un’altra URL (prima “Moved temporarily“)
#### 4xx errori client
- `400 Bad Request` → apparente errore del client
- `401 Unauthorized` → è richiesta l’autenticazione
- `403 Forbidden` → la richiesta conteneva dati validi ed è stata compresa dal server, ma l’azione è proibita
- `404 Not Found` → la risorsa non è stata trovata ma potrebbe essere disponibile in futuro
- `405 Method Not Allowed` → il metodo di richiesta non è 