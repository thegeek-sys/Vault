---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Il tipo $\verb|error|$
`error` è un’interfaccia predefinita in Go che implementa il metodo `Error() string`.

Le funzioni fallibili restituiscono `error` come **ultimo** valore

>[!example] Esempio
>Si verifica sempre l’errore subito dopo la chiamata
>```go
>func Divide(a, b  float64) (float64, error) {
>	if b==0 {
>		return 0, errors.New("divisione per zero")
>	}
>	return a/b, nil
>}
>
>func main() {
>	result, err := Divide(10, 0)
>	// controlla se c'è un errore
>	if err != nil {
>		fmt.Println("Error: ", err)
>		return
>	}
>	fmt.Println("Result: ", result)
>}
>```

---
## La tipologia dell’errore
Il tipo specifico (una `struct` o variabile `error`) di errore indica la causa o la natura del fallimento (es. `*os.PathError`), permettendo un controllo logico specifico

E’ possibile controllarlo con `errors.Is` (uguaglianza) o `errors.As` (conversione)

>[!example] Contorllo sul tipo di errore
>```go
>var ErrInvalidInput = errors.New("input non valido")
>
>func Process(data string) error {
>	if data == "" {
>		// restituisco il valore specifico ErrInvalidInput
>		return ErrInvalidInput
>	}
>	return nil
>}
>
>func main() {
>	err := Proces("")
>	
>	if errors.Is(err, ErrInvalidInput) {
>		fmt.Println("Errore logico: input mancante")
>	} else if err != nil {
>		fmt.Println("Errore generico: ", err)
>	}
>}
>```

>[!example] Utilizzo di struct
>```go
>type AuthError struct {
>	User string
>	Code int
>	Msg  string
>}
>
>// implementazione del metodo Error() per l'interfaccia error
>func (e *AuthError) Error() string {
>	return fmt.Sprintf("Auth fallita per %s: %s (codice %d)", e.User, e.Msg, e.Code)
>}
>
>func Authenticate(user string) error {
>	// in caso di errore specifico
>	return &AuthError{User: user, Code: 401, Msg: "Credenziali non valide"}
>}
>
>func main() {
>	err := Authenticate("ospite")
>	var authErr *AuthError
>	
>	// errors.As tenta di assegnare l'errore a 'authErr' se il tipo corrisponde
>	if errors.As(err, &authErr) {
>		fmt.Println("Autenticazione fallita:")
>		fmt.Printf("Utente: %s\n", authErr.User)
>		fmt.Printf("Codice: %d\n", authErr.Code)
>	} else if err != nil {
>		fmt.Println("Errore generico: ", err)
>	}
>}
>```