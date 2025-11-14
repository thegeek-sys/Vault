---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Composizione
Go non ha l’ereditarietà di classe, ma favorisce il riuso del codice tramite la **composizione** la quale evita la complessità dell’ereditarietà multipla.

Per farlo, si incorpora una `struct` all’interno di un’altra:
- embedding → inserire una `struct` senza nome di campo
- promozione → i metodi della struct interna sono promossi dalla struct esterna
- simula il riutilizzo senza ereditarietà di tipo

```go
type Logger struct {
	LogLevel string
}
func (l Logger) Log(msg string) {
	// logica di logging
}

// server 'ha' un Logger (composizione)
type Server struct {
	Logger // embedding
	Host string
}

func main() {
	s := Server{Logger: Logger{LogLevel: "INFO"}}
	
	// il metodo Log è ora accessibile direttamente da 's'
	s.Log("Avvio del server")
}
```