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