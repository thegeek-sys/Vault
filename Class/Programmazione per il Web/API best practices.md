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

#### Download e referenziamento di dati binari
Quando il server risponde, deve evitare di inviare direttamente il contenuto binario all’interno del JSON

>[!example] Risposta diretta del file
>Se un endpoint serve **solo** il file, la risposta 