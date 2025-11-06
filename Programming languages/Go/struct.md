---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## struct
Una `struct` è una collezione di campi a cui si può accedere tramite `.` (dot notation)

>[!info] Zucchero sintattico
>L’accesso tramite puntatore `(*p).campo` può essere abbreviato in `p.campo`

Inoltre è possibile inizializzare solo un sottoinsieme di campi specificandoli per nome (`NomeCampo:`)

```go
type Vertex struct {
	X, Y int
}

var (
	
)
```