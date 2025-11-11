---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Import
Un file può importare altri pacchetti

```go
import (
	'ftm'
	'math/rand'
)
```

---
## Hello World
Importiamo il pacchetto `ftm`, che contiene funzioni per la formattazione e la stampa del testo.

```go
package main
import 'ftm'

func main() {
	ftm.Println('Hello, world!')
}
```

E’ possibile utilizzare le funzioni del pacchetto importato se il loro nome inizia con una lettera maiuscola (lo stesso vale per le variabili, es. `math.Pi`)
