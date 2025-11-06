---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Tipizzazione e assegnazione di variabili

```go
// stessa sintassi per 'const' invece di 'var'
var i int

// stessa sintassi per 'const'
var i int = 3

// tipo inferito; funziona per 'const' MA è senza tipo (untyped)
var i = 3

// non si può usare con 'const'
i := 3
```

Le costanti non possono essere dichiarate usando la sintassi `:=`

### Inferenza di tipo
Senza specificare un tipo esplicito (utilizzando la sintassi `:=` o l’espressione `var=`), Go inferisce il tipo

```go
i := 42           // int
f := 3.1415       // float64
g := 0.867 + 0.5i // complex128
```
---
## Tipi di dati di base

```go
bool
string
int int8 int16 int32 int64
uint uint8 uint16 uint32 uint64 uintptr
byte // alias per uint8
rune // rappresenta un codepoint in Unicode
float32 float64
complex64 complex128
```

I tipi `int`, `uint`, `uintptr` sono solitamente a 32 bit sui sistemi a 32 bit e a 64 bit sui sistemi a 64 bit

### Valori di default
- `0` → tipi numerici
- `false` → tipo booleano
- `""` (stringa vuota) → stringa

---
## Costanti numeriche
Le costanti senza tipo (`untyped`) assumono un solo tipo quando vengono usate. Possono essere usate infatti in contesti che richiedono tipi numerici diversi (come `int` o `float`), perché il valore rie