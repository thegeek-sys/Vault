---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Valori di ritorno multipli

```go
package main
import "fmt"

func divide(x int, y int) (int, int) {
	return x/y, x%y
}

func main() {
	fmt.Println(divide(7, 3)) // out:: 2 1
}
```

### Valori di ritorno nominati

```go
package main
import "fmt"

func divide(x int, y int) (int, int) {
	div = x/y
	mod = x%y
	return // restituisce div e mod
}

func main() {
	fmt.Println(divide(7, 3)) // out:: 2 1
}
```

---
## Call-by value
Go usa sempre il **call-by value** per cui le modifiche interne non sono visibili all’esterno

>[!tip]
>Usato per tipi base: `int`, `string`, `struc` piccole

```go
func increment(x int) {
	x++ // modifica solo la copia locale
}
func main() {
	a := 5
	increment(a)
	// a non cambia
}
```

---
## Call-by reference
Utilizzando la call-by reference viene passata la copia dell’indirizzo di memoria (`*T`) e dunque la funzione modifica il valore originale tramite l’indirizzo

>[!tip]
>Usato per: `struct` grandi o quando è necessaria la modifica

```go
func scale(v *int) {
	*v *= 10 // dereferenzia e modifica l'originale
}
func main() {
	a := 5
	scale(&a) // passa l'indirizzo di a
	// a è diventato 50
}
```

Per quanto riguarda `slice`, `map` e `channel`, sono tipi di riferimento, dunque la loro struttura è copiata, ma il dato sottostante è condiviso (la copia contiene un puntatore alla struttura dati)

