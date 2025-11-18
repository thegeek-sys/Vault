---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Il server http integrato di Go
Go include una libreria HTTP completa nello standard package `net/http`
- `http.Handler` → interfaccia usata per gestire le richieste
- `http.HandleFunc` → funzione per associare un percorso a una funzione gestore (hook)
- `http.ListenAndServe` → avvia il server HTTP

Il gestore di richieste la firma:
```go
func handler(w http.ResponseWriter, r *http.Request)
```

>[!example] Servizio “pari o dispari”
>Creiamo un endpoint semplice che genera un numero casuale e ne restituisce la parità
>- `http.ResponseWriter(x)` → usato per scrivere la risposta inviata al server (corpo della risposta)
>- `w.Header().Set(...)` → utilizzato per impostare gli header della risposta (es. `Content-Type`); deve essere chiamato prima di scrivere il corpo!
>- `fmt.Fprintf(w, ...)` → funzione utile che accetta un `io.Writer` (come `w`) per scrivere la risposta
>
>```go
>func evenRandomNumber(w http.ResponseWriter, r *http.Request) {
>	// imposta l'header prima di scrivere il corpo
>	w.Header().Set("Content-Type", "text/plain")
>	
>	p:=rand.Int()
>	if p%2 == 0 {
>		fmt.Fprintf(w, "%d is even", p)
>	} else {
>		fmt.Fprintf(w, "%d is odd", p)
>	}
>}
>```

---
## Lettura e ricezione di dati

>[!info] Input $\verb|*http.request|$
Il parametro `r` (di tipo `*http.Request`) contiene tutte le informazioni inviate dal client. 

### Lettura
#### Query string parameters
I query string parameters sono parte dell’URL e vengono usati per passare dati semplici o opzioni di filtraggio. Si usano per accettare dati da un client tramite l’URL (metodo `GET`).

Accesso ai parametri:
1. la richiesta `r` contiene l’URL: `r.URL` (di tipo `*url.URL`)
2. si usa `.Query()` per ottenere una mappa dei parametri (di tipo `url.Values`, mappa di stringhe)
3. si usa `Get("nome_parametro")` per estrarre il valore

>[!example] Url: $\verb|http://localhost:8090/?name=John+Doe|$
>```go
>func hi(w http.Responsewriter, r *http.Request) {
>	// estrae il valore del parametro "name"
>	// si tratta di una mappa che associa i nomi dei parametri
>	// ai loro valori (url.Values)
>	name := r.URL.Query().Get("name")
>	
>	w.Header().Set("Content-Type", "text/plain")
>	
>	// stampa il saluto, utilizzando il nome ricevuto
>	fmt.Fprintf(w, "Hi %s!", name)
>}
>```

#### Path parameters
Questi parametri sono integrati nel percorso dell’URL (`/users/101`)

| Componente                        | Package       | Descrizione                                                                                                                                     |
| --------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Standard Library (`net/http`)     |               | il gestore deve analizzare manualmente la stringa `r.URL.Path` (e.g. usando `strings.Split`) per estrarre i segmenti variabili. Non consigliato |
| Router esterno (es. `Gorill Mux`) | `mux.Vars(r)` | restituisce una mappa (`map[string]string`) contiene i parametri estratti dal path. Approccio standard                                          |
>[!example] Esempio manuale
>Suddivide il percorso in segmenti per estrarre l’ID
>```go
>segments := string.Split(r.URL.Path, "/")
>```

### Recezione: post body
Per ricevere dati strutturati (come testo o JSON) inviati con richiesta `POST` (o `PUT`), è necessario leggere il corpo della richiesta. In particolare il parametro `w` (di tipo `http.ResponseWriter`) è l’interfaccia usata per costruire e inviare la risposta al client.

Lettura del body:
1. il  corpo della richiesta è accessibile tramite `r.Body` (di tipo `io.ReadCloser`)
2. usiamo `io.ReadAll(r.Body)` per leggere tutto il contenuto di un array di byte `[]byte`

>[!example] Dati non strutturati
>```go
>func hi(w http.ResponseWriter, r *http.Request) {
>	// leggere l'intero corpo della ricehista come []byte
>	body, _ := io.ReadAll(r.Body)
>	
>	// converte il body (si assume sia il nome in plain text) in stringa
>	name := string(body)
>	
>	// header devono essere impostati prima di scrivere il
>	// corpo della risposta
>	w.Header().Set("Content-Type", "text/plain")
>	
>	// imposta lo statusc code HTTP. default 200 ok
>	w.WriteHeader(htt.StatusCreated) // 201
>	fmt.Fprintf(w, "Hi %s!", name)
>	// in alternativa posso usare w.Writer([]byte)
>	
>	// in un server reale si deve sempre controllare l'errore di
>	// io.ReadAll e l'header Content-Type della richiesta per dati
>	// strutturati
>}
>```
>
>>[!info]
>>Si testa questa funzionalità tipicamente con:
>>```bash
>>curl -d 'John Doe' http://localhost:8090
>>```

---
## Riassunto e avvio
Tutti i gestori devono essere registrati nel *multiplexer* HTTP di Go:
- registrazione → `http.HandleFunc(path, handler_function)`
- avvio → `http.ListenAndServer(":porta", nil)`

```go
func main() {
	// registra la funzione gestore per il percorso radice
	http.HandleFunc("/", hi)
	
	fmt.Println("Starting web server at http://localhost:8090")
	// avvia il server sulla porta 8090
	http.ListenAndServer(":8090", nil)
}
```

---
## Gestione di dati strutturati: json
Nel web moderno, JSON è il formato standard per lo scambio di dati. Go usa il package `encoding/json` per mappare i dati tra le `struct` di Go e le stringhe di JSON

### Leggere JSON della richiesta (decoding)
Per leggere un corpo JSON in una `struct` Go, si usa `json.NewDecoder`

```go
type UserInput struct {
	Name string 'json:"name"'
	Age  int    'json:"age"'
}
// ...
var input UserInput
err := json.NewDecorder(r.Body).Decode(&input)
// ...
```

### Scrivere JSON nella risposta
Per inviare una `struct` Go come risposta JSON, si usa `json.Marshal` o `json.NewEncoder`

```go
type UserResponse struct {
	Message string 'json:"status"'
}
// ...
response := UserResponse{Message: "Success"}
w.Header().Set("Content-Type", "application/json")
json.NewEncoder(w).Encode(response)
```