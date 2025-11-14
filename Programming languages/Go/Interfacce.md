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

```