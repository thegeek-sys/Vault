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

### Lettura: query string parameters
Per accettare dati da un client tramite l’URL (metodo `GET`), si usano i **query string parameters**

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

### Ricezione: post body
Per ricevere dati strutturati (come testo o JSON) inviati con richiesta `POST`, è necessario leggere il corpo della richiesta.

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
>	w.Header().Set("Content-Type", "text/plain")
>	fmt.Fprintf(w, "Hi %s!", name)
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
