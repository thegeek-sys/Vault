---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Definizione
Un’interfaccia definisce un **insieme di recipe di metodi**. Un tipo implementa un’interfaccia se implementa tutti i metodi definiti nell’interfaccia, e la loro implementazione è implicita (non serve nessuna keyword)

---
## Interfaccia $\verb|Stringer|$
Un esempio comune di interfaccia della libreria standard

```go
// fmt/print.go:62
type Stringer interface {
	String() string
}
```

### Struct che la implementa
Qualsiasi tipo che ha il metodo `String() string` implementa implicitamente `Stringer`

```go
type Vertex struct {
	X int
	Y int
}

// Vertex implementa Stringer perchè ha il metodo String()
func (v *Vertex) String() string {
	return fmt.Sprintf("(%d, %d)", v.X, v.Y)
}

// funzione che accetta un'interfaccia Stringer
func printSomething(s, fmt.Stringer) {
	fmt.Println(s.String())
}

// v := &Vertex{10,20}
// printSomething(v) // out: (10, 20)
```

---
## Interfaccia $\verb|Writer|$
L’interfaccia `io.Writer` è fondamentale in Go per l’I/O

```go
// io/io.go:96
type Writer interface {
	Write(p []byte) (n int, err error)
}
```

### Implementazioni comuni
Qualsiasi cosa che può accettare byte implementa `Writer`, ad esempio:
- `os.Stdout` → lo standard output della console
- `bytes.Buffer` → un buffer di byte di memoria
- `os.File` → qualsiasi file su disco
- `net.Conn` → una connessione di rete (socket)

Le interfacce consentono il **polimorfismo** e il **disaccoppiamento**, permettendo a funzioni come `fmt.Fprintf` di lavorare con qualsiasi `Writer`

