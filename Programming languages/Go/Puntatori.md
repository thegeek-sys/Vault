---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Puntatori
Un puntatore contiene l’indirizzo di memoria di un valore, ad esempio il tipo `*T` è un puntatore a un valore di tipo `T`.

Il valore di default è `nil`, e bisogna distinguere `*` e `&`:
- `&` (operatore di indirizzo) → genera un puntatore a una variabile
- `*` (operatore di deferenziazione) → legge o importa il valore a cui punta

A differenza di C, Go non ha aritmetica sui puntatori

```go
package main
import 'fmt'

func main() {
	var p *int
	i := 42
	p = &i          // p punta all'indirizzo di i
	fmt.Println(p)  // stampa l'indirizzo di memoria di i
	fmt.Println(*p) // stampa il valore di i tramite il puntatore p
	*p = 21         // imposta i a 21 tramite il puntatore p
	fmt.Println(i)  // stampa 21
}
```