---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
![[struct#Metodi su struct]]

---
## Metodi su tipi personalizzati
I metodi possono essere definiti su qualsiasi tipo (ad eccezione dei tipi puntatore o di interfaccia)

### Tipi di base con metodi

```go
type Latitude float64

// metodo definito su Latitude
func (lat *Latitude) IsValid() bool {
	// *lat è il valore
	return lat != nil && *lat >= -90 && *lat <= 90
}

type Longitude float64

// metodo definito su Latitude
func (lon *Longitude) IsValid() bool {
	// *lat è il valore
	return lon != nil && *lon >= -180 && *lon <= 180
}
```

### Struct che usato tipi con metodi
I metodi consentono di riutilizzare la logica di validazione nei tipi più complessi

```go
type Point2D struct {
	Latitude Latitude
	Longitude Longitude
}

// metodo definito su Point2D
func (p Point2D) IsValid() bool {
	// chiama i metodi IsValid() definiti sui campi
	return p.Latitude.IsValid() && p.Longitude.IsValid()
}

func (p Point2D) IsZero() bool {
	return p.Latitude==0 && p.Longitude==0
}
```