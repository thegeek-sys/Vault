---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Packages
Le funzioni Go sono raggruppate in pacchetti e ogni pacchetto è composto da uno o più file nella stessa directory. Il pacchetto è dichiarato all’inizio del file

```go
package main
```

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