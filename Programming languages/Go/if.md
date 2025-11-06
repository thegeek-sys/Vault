---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Regole e scope
Come per il `for`, anche nell’`if` non è necessaria nessuna parentesi `()` per le condizioni ma sono sempre necessarie le graffe `{}`.

Un `if` in Go ha la particolarità che può iniziare con un’istruzione eseguita prima della condizione (*short statement*). Inoltre per quanto riguarda lo scoping, le variabili dichiarate nello short statement sono visibili solo all’interno del blocco `if` e dei successivi `else` ed è possibile utilizzarli anche all’interno degli `else if`.
Gli short statement vengono solitamente usati per evitare di sporcare lo statement 

```go
func pow(x, n, lim float64) float64 {
	// v dichiarata e inizializzata qui
	if v := math.Pow(x, n); v < lim {
		return v
	}
	return lim // v non è visibile fuori dall'if
}
```