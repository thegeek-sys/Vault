---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Packages
Le funzioni Go sono raggruppate in pacchetti e ogni pacchetto è composto da uno o più file nella stessa directory. Il pacchetto è dichiarato all’inizio del file

Un programma Go è una composizione di **pacchetti**.

```go
package main
```

La funzione principale è `main()` all’interno del pacchetto `main`. I pacchetti vengono importati utilizzando il loro percorso ed ognuno di loro deve risiedere in una directory dedicata

### Module
Un moduleè una collezione di uno o più pacchetti che server come unità di gestione delle dipendenze e controllo di versione, definito dal file `go.mod`

![[Tips & tricks#Creare e definire il modulo]]

### Utilizzare i pacchetti

```go
package main

import (
	"fmt"
	"math/rand"
)

func main() {
	fmt.Println("Il mio numero preferito è ", rand.Intn(10))
}
```

---
## Libreria standard
Go fornisce una libreria standard con pacchetti comuni, ma è anche possibile creare pacchetti all’intendo del proprio progetto oppure usare pacchetti esterni

### Creare pacchetti all’interno di un progetto
1. All’interno della directory del progetto, `go mod init nome-modulo` crea un modulo (`go.mod` e `go.sum`)
2. Ora è possibile creare i propri sotto-moduli per i pacchetti
3. Bisogna ora importare i sotto-moduli usando il percorso completo

>[!info] Struttura
>```
>main.go
>go.mod
>go.sum
>package1/
>	functions.go
>	other-things.go
>	subpackage/
>		file.go
>```
>
>>[!example] Pacchetto interno
>>Contenuto di `package1/functions.go`
>>```go
>>package package1
>>
>>// dobbiamo usare la lettera maiuscola come prima
>>// lettera per rendere questa funzione pubblica
>>// altrimenti sarà disponibile solo all'interno di package1
>>func Dummy() {}
>>
>>// questa funzione è privata
>>func internalFunction() {}
>>```
>
>
>>[!example] Contenuto di `main.go`
>>```go
>>package main
>>
>>import "nome-modulo/package1" // importa il pacchetto interno
>>
>>func main() {
>>	// chiama la funzione pubblica Dummy
>>	package1.Dummy()
>>	// package1.internalFunction() // errore
>>}
>>```

