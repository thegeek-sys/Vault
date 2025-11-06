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
	v1 = Vertex{1, 2} // ha tipo Vertex
	v2 = Vertex{X: 1} // Y:0 è implicito
	v3 = Vertex{}     // X:0 e Y:0 impliciti
	p = &Vertex{1, 2} // ha tipo *Vertex (puntatore a Vertex)
)
```

---
## struct anonime
Le `struct` possono essere dichiarate senza un nome di tipo esplicito, utili se è necessaria una `struct` da usare solo localmente

```go
a := struct{
	i int
	b bool
}{1, true}
```