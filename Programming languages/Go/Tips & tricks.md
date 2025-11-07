---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Tips & tricks
### Importazioni fattorizzate

```go
import "fmt"
import "math"

// uguale a 

import (
	"fmt"
	"math"
)
```

### Variabili fattorizzate
La stessa cosa funziona anche per le variabili globali

```go
import "complex"

var (
	ToBe bool = false
	MaxInt uint64 = 1<<64-1
	z complex128 = complex128(complex.Sqrt(-5 + 12i))
)
```

### Specificare il tipo per parametri multipli

```go
x int, y int
// uguale a
x, y int
```

---
## Inizializzazione e struttura del progetto
La toolchain di Go utilizza i *modules* per gestire dipendenze e versioni

### Creare e definire il modulo
Il comando `go mod` è fondamentale per inizializzare e gestire un progetto

| Comando              | Descrizione                                                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `go⠀mod⠀init⠀<path>` | inizializza un nuovo modulo G<br>crea il file `go.mod` nella cwd, definendo il percorso radice del modulo (es. url di repo github)           |
| `go⠀mod⠀tidy`        | pulisce e aggiorna<br>aggiunge eventuali dipendenze mancanti (trovate nel codice) e rimuove quelle inutilizzare dal file `go.mod` e `go.sum` |
| `go⠀mod⠀download`    | scarica tutti i moduli richiesti in `go.mod` nella cache locale                                                                              |

### Struttura standard
I progetti Go seguono una convenzione semplice:
- source file → i file in Go finiscono con `.go`
- package main → contiene la funzione `main()` ed è il punto di ingresso per un programma eseguibile
- package (custom name) → contiene librerie e logica di business riutilizzabile

>[!example] Esempio
>1. crea la cartella → `mkdir myproject && cd myproject`
>2. inizializza il modulo → `go mod init example.com/myproject`
>3. crea il file principale → `touch main.go`

### Run ed esecuzione rapida
Per eseguire rapidamente il codice senza produrre un eseguibile finale si usa uno dei seguenti comandi

| Comando          | Descrizione                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `go⠀run⠀main.go` | compila ed esegue il codice sorgente specificato; utile per testare rapidamente durante lo sviluppo |
| `go⠀run⠀.`       | compila ed esegue il package `main` nella directory corrente                                        |

### Compilazione e build
Per creare un eseguibile binario e distribuibile si usa uno dei seguenti comando

| Comando                  | Descrizione                                                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `go⠀build⠀.`             | compila il package corrente (o `main` se eseguibile) e genera un file binario nella directory corrente, chiamato come il modulo o la directory |
| `go⠀build⠀-o app_name⠀.` | compila e specifica il nome del file di output (es. `app_name`)                                                                                |
| `go⠀install⠀.`           | compila e install l’eseguibile nella cartella binari predefinita di Go (`$GOPATH/bin` o `$GOBIN`)                                              |

### Test
Il framework di testing è integrato nella toolchain

| Comando               | Descrizione                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `go⠀test`             | esegue i test (file con suffisso `_test.go`) nella directory corrente |
| `go⠀build⠀test⠀./...` | esegue i test in tutte le sottodirectory                              |
| `go⠀test⠀-v`          | esegue i test in modalità verbosa                                     |

