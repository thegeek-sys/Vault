---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## map
Le mappe servono a mappare chiavi a valori. Anche qui il valore di default è `nil` (e non si possono aggiungere chiavi ad una mappa `nil`) e per questo si deve usare la funzione `make` per inizializzare una mappa

```go
package main
import 'fmt'

var m map[string]string // m è nil, non si possono aggiungere elementi

func main() {
	m = make(map[string]string) // chiave string, valore string
	m['Bell labs'] = 'cooked'
	fmt.Println(m['Bell labs']) // cooked
}
```

---
## Map literals
Sono simili a `struct` literals, ma le chiavi (e i valori) sono obbligatori

```go
type Vertex struct {
	Lat, Long float64
}

var m = map[string]Vertex{
	"Bell labs": Vertex{
		40.68433, -74.39967
	},
	"Google": Vertex{
		37.42202, -122.08408
	}
}
```

---
## Modificare le mappe
- inserimento/aggiornamento (upsert) → `m[key] = elem`
- lettura → `elem := m[key]`
- cancellazione → `delete(m, key)` (fallisce silenziosamente se la chiave è assente)
- test di presenza → `elem, ok := m[key]`
	- `ok=true` → se chiave è presente
	- `elem=0` → se chiave non è presente

