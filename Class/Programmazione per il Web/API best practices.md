---
Class: "[[Programmazione per il Web]]"
Related:
  - "[[REST]]"
---
---
## Campi opzionali
In OpenAPI, il campo `required` definisce le regole di validazione del payload e non il comportamento logico dell’API

La chiave `required` in uno schema determina se un campo deve essere presente nel payload in ingresso (es. nel `requestBody` di una `POST` o `PUT`)


| Contesto              | Campo id (`readOnly: true`)                | Regola di `required`                                                             |
| --------------------- | ------------------------------------------ | -------------------------------------------------------------------------------- |
| `POST /fountains`     | l’ID non può essere inviato                | deve essere omesso nel `requestBody`                                             |
| `PUT /fountains/{id}` | l’ID non deve essere modificato dal client | deve essere inviato (per la sostituzione completa) ma il server lo ignora/valida |
| `GET /fountains/{id}` | l’ID è restituito                          | deve essere presente nello schema di risposta                                    |

I campi opzionali (quelli non elencati in `required: []`) e i campi in sola lettura (`readOnly: true`) dovrebbero essere gestiti in modo specifico negli esempi

>[!example] Esempio di `requestBody` (POST)
>- L’esempio non deve includere il campo id

>[!example] Esempio di `responseBody` (201 Created)
>L’esempio deve includere il campo id

---
## URIS

>[!warning]
>Le URI rappresentano risorse, non azioni

Per questo motivo nelle URI bisogna usare i sostantivi (identificano oggetti) e mai includere verbi (l’azione è definita dal metodo HTTP)

| Metodo   | Esempio (corretto)      | Esempio (errato)              |
| -------- | ----------------------- | ----------------------------- |
| `GET`    | `/managed-devices/{id}` | `/get-managed-device/{id}`    |
| `PUT`    | `/managed-devices/{id}` | `/update-managed-device/{id}` |
| `POST`   | `/users`                | `/create-user`                |
| `DELETE` | `/managed-devices/{id}` | `/remove-managed-device/{id}` |
Per convenzione si usano sostantivi *singolari* per le singole risorse e *plurali* per le collezioni

>[!example]
>- singolo: `/users/admin`
>- collezione: `/users`

>[!bug] GET e DELETE
>Non bisogna definire un corpo nelle richieste `GET` e `DELETE`. Sebbene `DELETE` possa tecnicamente avere un corpo (per lo standard HTTP), è sconsigliato in REST per evitare comportamenti indefiniti e complicazione con l’idempotenza

---
## Struttura e gerarchia
Per la **gerarchia** si usa `/` per esprimere una gerarchia logica nella relazione tra le risorse; ciò aiuta l’API a crescere in modo orinato

>[!example] Esempio di annidamento logico
>```
>/users/{userid}/devices/{deviceid}
>```
>
>Il dispositivo `{deviceid}` è una risorsa contenuta all’interno della collezione di dispositivi dell’utente `{userid}`

>[!warning]
>Potenzialmente il link con o senza slash possono essere diversi

---
## Gestione degli errori e filtri
### Codici di stato HTTP
Molti errori delle API possono essere mappati direttamente a specifici codici di stato HTTP

>[!example]
>- parametri mancanti o valori non validi: 400 **Bad Request**
>- risorsa non trovata: 404 **Not Found**
>- successo: 200 **OK**, 201 **Created**, 204 **No Content**

Se necessario è possibile fornire dettagli aggiuntivi riguardo l’errore in un corpo di risposta comune

### Query per filtri e paginazione
Per filtrare e impaginare i risultati bisogna usare i componenti *query string* e MAI usare il body o l’URI del percorso per i filtri

>[!info]
>La collezione è unica i parametri la limitano

>[!example]
>```
>/managed-devices/?region=USA
>```

---
## Dati binari e corpo della richiesta
### Dati binari
Non è consigliabile incorporare grandi dati binari (es. foto) direttamente all’interno di JSON per motivi di performance è piuttosto preferibile:
- invio ($\text{Client} \to \text{Server}$)
	- usa `multipart/form-data` per inviare dati strutturali (JSON) e il file in un’unica richiesta
	- carica il file in una API dedicata (es. `POST /images/`) e poi usa l’URL/ID dell’immagine nella richiesta JSON principale
- ricezione ($\text{Server}\to \text{Client}$)
	- restituisci un JSON con l’informazione strutturata e l’URL completo per accedere al file binario in un endpoint separato (es. `/images/{id}`)

#### Upload di dati strutturati + file (multipart)
Se devi inviare dati JSON insieme a un file binario, l’approccio standard è utilizzare `multipart/form-data`

>[!example] Aggiungere una nuova fontana (JSON) e caricarne contemporaneamente la foto (file)
>**Corpo della richiesta**:
>- media type: `multipart/form-data`
>- part 1 (JSON): `Content-Disposition: name="fountain_data"` (contenente il JSON)
>- part 2 (file binario): `Content-Disposition: name="photo"` (contiene i byte binari dell’immagine)

>[!example] Esempio con CURL
>Il comando `curl` semplifica enormemente l’uso di `multipart/form-data` grazie all’opzione `-F` (o `--form`)
>
>Quando si usa `-F`, `curl` esegue automaticamente tre azioni cruciali:
>1. imposta l’intestazione `Content-Type: multipart/form-data`
>2. genera e imposta un **valore di boundary** unico per la richiesta
>3. formatta il corpo della richiesta, separando correttamente i dati
>
>Scenario pratico: aggiungere una fontana (JSON) e caricarne la foto
>Il comando `curl` richieste un parametro `-F` per ogni “parte” che deve essere inviata
>
>| Parte            | Opzione cURL                                          | Descrizione                                                         |
>| ---------------- | ----------------------------------------------------- | ------------------------------------------------------------------- |
>| Dati strutturati | `-F "fountain_data=@data.json;type=application/json"` | Carica il contenuto di `data.json` e ne specifica il `Content-Type` |
>| File binario     | `-F "photo=@image.jpg;type=image/jpeg"`               | Carica il file binario `image.jpg`                                  |
>
>```
>curl -X POST https://api.nasoniroma.id/v1/fountains \
>	-F "fountain_data=@data.json;type=application/json" \
>	-F "photo=@image.jpg;type=image/jpeg"
>```

#### Download e referenziamento di dati binari
Quando il server risponde, deve evitare di inviare direttamente il contenuto binario all’interno del JSON

>[!example] Risposta diretta del file
>Se un endpoint serve **solo** il file, la risposta non è JSON
>```
>GET /images/fountain_123.jpg
>```
>
>Risposta HTTP:
>```
>Status: 200 OK
>Content-Type: image/jpeg
>Body: byte dell'immagine
>```

>[!example] Referenziamento nel JSON
>Quando si listano le fontane o si recupera una singola risorsa, il JSON deve contenere un riferimento (URL o ID) al file
>
>```
>{
>	"id": 1,
>	"state": "good",
>	"latitude": 41.89025
>	"longitude": 12.49237
>	"photo_url": "/api/v1/images/fountain_123.jpg" // riferimento al file
>}
>```
>
>Il vantaggio sta nel fatto che il client decide se e quando scaricare l’immagine, ottimizzando i tempi di caricamento della lista principale

---
## Gestione delle sotto-collezioni

>[!info] Principio
>Non scambiare nome della collezione e elemento

Se una risorsa (o un’azione) ha effetto solo sull’utente autenticato (es. mutare Bob per il mio utente), bisogna trattare l’azione come una collezione di risorse relative all’utente autenticato

```
/me/muted/{user_id}
```

Questo approccio garantisce che le azioni siano **idempotenti**, che l’operazione di “unmuting” sia nativamente gestita dal `DELETE`, e che la risorsa sia chiaramente *per-utente-autenticato*

---
## Aggiornamento delle risorse: PUT vs PATCH
Quando si tratta di modificare una risorsa esistente, si usano principalmente due metodi HTTP con semantiche molto diverse.

### PUT: sostituzione completa

| Caratteristica | Descrizione                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------- |
| Scopo          | Sostituire la risorsa bersaglio con il contenuto  completo inviato nel corpo della richiesta                        |
| Idempotenza    | Eseguire la stessa richiesta PUT più volte produce lo stesso stato finale sul server                                |
| Dati richiesti | L’intera rappresentazione della risorsa. Se ometti un campo, questo dovrebbe essere eliminato o azzerato dal server |
| Tipico uso     | Modifiche complete, come la riscrittura di un documento                                                             |
### PATCH: sostituzione parziale

| Caratteristica | Descrizione                                                                   |
| -------------- | ----------------------------------------------------------------------------- |
| Scopo          | Applicare una serie di modifiche parziali alla risorsa bersaglio              |
| Idempotenza    | Una richiesta PATCH è idempotente solo se il server la implementa per esserlo |
| Dati richiesti | Solo i campi da modificare. I campi omessi rimangono invariati                |
| Tipico uso     | Modifiche mirate, come cambiare solo lo stato o il flag di una risorsa        |
