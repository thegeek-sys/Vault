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

### Costanti numeriche
Le costanti senza tipo (`untyped`) assumono un solo tipo quando vengono usate. Possono essere usate infatti in contesti che richiedono tipi numerici diversi (come `int` o `float`), purché il valore rientri nel range.

```go
const (
	// non hanno un tipo specifico
	untypedInt = 1     // es. int o float64
	untypedFloat = 1.1 // es. float64
)

func needInt(x int) int {
	return x*10+1
}
func needFloat(x float64) float64 {
	return x*0.1
}

func main() {
	fmt.Println(needInt(untypedInt))   // ok
	fmt.Println(needFloat(untypedInt)) // ok
	fmt.Println(needInt(untypedFloat)) // errore non possibile convertire
									   // float a int senza explicit casting
}
```

#### Casting
L’espressione `T(v)` converte il valore `v` al tipo `T` e risulta necessario per convertire tra tipi numerici.

```go
i := 42
f := float64(i) // i (int) convertito in f (float64)
u := uint(f)    // f (float64) convertito in u (uint)
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

### Rune
La `rune` rappresenta in senso proprio un carattere, che è diverso dalla vera e propria lunghezza di una stringa

>[!example]
>
>```go
>package main
>import (
>	'fmt'
>	'uncode/utf8'
>)
>
>func main() {
>	s := 'Hello, world! 🌍'
>	ftm.Println("Len (byte): ", len(s))                    // 17
>	ftm.Println("Rune count: ", utf8.RuneCountInString(s)) // 14
>}
>```

Infatti il carattere 🌍 (U+1F30D) viene codificato in UTF-8 usando 4 byte, ma rappresenta un solo Rune dato che questo è il carattere logico che Go usa per l’iterazione e il conteggio


| Categoria Rune | Intervallo code point (hex) | Byte per rune (in UTF-8) | Esempio | Spiegazione                        |
| -------------- | --------------------------- | ------------------------ | ------- | ---------------------------------- |
| 1 byte         | `U+0000` a `U+007F`         | 1 byte                   | `A 1 !` | caratteri ASCII di base            |
| 2 byte         | `U+0080` a `U+07FF`         | 2 byte                   | `é ñ £` | caratteri latini estesi, valute    |
| 3 byte         | `U+0800` a `U+FFFF`         | 4 byte                   | `🔥 🥰` | emoji, pittogrammi, caratteri rari |

### Valori di default
- `0` → tipi numerici
- `false` → tipo booleano
- `""` (stringa vuota) → stringa

