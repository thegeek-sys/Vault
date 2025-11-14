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

---
## Metodi su struct
I metodi sono funzioni con un argomento receiver speciale

### Receiver per valore (copia)

```go
type Vertex struct {
	X int
	Y int
}

// receiver per valore: opera su una copia della struct
func (v Vertex) Equal(other Vertex) bool {
	return v.X == other.X && v.Y == other.Y
}

// vtx := Vertex{1,2}
// if vtx.Equal(Vertex{2,3}) {...}
```

### Receiver per puntatore

```go
type Vertex struct {
	X int
	Y int
}

// receiver per puntatore: opera sulla struct originale (può modificare)
func (v *Vertex) Scale(factor int) {
	v.X *= factor
	v.Y *= factor
}

// vtx := &Vertex{1,2}
// vtx.Scale(10) // ora vtx.X e vtx.Y sono moltiplicati per 10
```