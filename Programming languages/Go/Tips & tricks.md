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
| `go⠀build⠀-o⠀app_name⠀.` | compila e specifica il nome del file di output (es. `app_name`)                                                                                |
| `go⠀install⠀.`⠀⠀⠀⠀⠀⠀⠀⠀   | compila e install l’eseguibile nella cartella binari predefinita di Go (`$GOPATH/bin` o `$GOBIN`)                                              |

### Test
Il framework di testing è integrato nella toolchain

| Comando               | Descrizione                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `go⠀test`             | esegue i test (file con suffisso `_test.go`) nella directory corrente |
| `go⠀build⠀test⠀./...` | esegue i test in tutte le sottodirectory                              |
| `go⠀test⠀-v`          | esegue i test in modalità verbosa                                     |

### Formattazione del codice
Go impone uno stile di formattazione standardizzato

| Comando    | Descrizione                                                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `go⠀fmt⠀.` | riformatta automaticamente tutti i file Go nella directory corrente in modo standardizzato, utilizzando lo stile canonico di Go |

### Analisi e linting
Strumenti integrati per l’analisi statistica del codice

| Comando                 | Descrizione                                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `go⠀vet⠀.`              | analizza il codice sorgente per identificare potenziali bug o costrutti sospetti, come errori comuni di formattazione di stringhe o assegnazioni inutili |
| `go⠀doc⠀<package/func>` | mostra la documentazione per un pacchetto o una funzione specifica                                                                                       |
### Altri strumenti utili
Comandi per la gestione e ispezione dell’ambiente

| Comando                 | Descrizione                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `go⠀get⠀<path/package>` | aggiunge un nuovo pacchetto esterno al file `go.mod` e lo scarica. |
| `go⠀version`            | mostra la versione di Go installata.                               |
| `go⠀env`                | stampa le variabili d'ambiente di Go (es. `$GOPATH`, `$GOBIN`).    |
### Aggiunta e aggiornamenti
Comandi che gestiscono le dipendenze esterne richieste dal codice

| Comando                        | Descrizione                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------- |
| `go⠀get⠀<path/package>`        | aggiunge una nuova dipendenza al progetto (o aggiorna una esistente) e la registra in `go.mod`. |
| `go⠀get⠀<path/package>@v1.2.3` | scarica e utilizza una specifica versione o un tag.                                             |
| `go⠀get⠀-u⠀./...`              | aggiorna tutte le dipendenze del modulo alla versione patch o minor più recente.                |
| `go⠀get⠀-u=patch⠀./...`        | aggiorna solo alle versioni patch più recenti (più sicuro).                                     |

### Ispezione e blocco ($\verb|go.sum|$)
Il file `go.sum` contiene gli hash crittografici per le dipendenze, garantendo l’integrità

| Comando           | Descrizione                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `go⠀list⠀-m⠀all`  | elenca tutte le dipendenze del modulo, incluse quelle transitive (di cui dipendono le tue dipendenze).                |
| `go⠀mod⠀vendor`⠀⠀ | crea una cartella `vendor/` contenente le copie locali di tutte le dipendenze, per build isolate o per reti limitate. |
| `go⠀mod⠀verify`   | verifica che i moduli scaricati nella cache Go corrispondano agli hash in `go.sum`.                                   |
### Pulizia
Per mantenere pulito l’ambiente del modulo

| Comando                        | Descrizione                                                                                                            |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `go⠀clean⠀-modcache` ⠀⠀⠀⠀⠀⠀⠀⠀⠀ | cancella la cache dei moduli (`$GOPATH/pkg/mod`), forzando il download di tutte le dipendenze alla prossima build/run. |
