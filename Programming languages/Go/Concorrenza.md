---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Concorrenza vs. parallelismo
### Concorrenza
La concorrenza riguarda la struttura del programma. Essa infatti permette di gestire molte cose contemporaneamente (passando da un compito all’altro).

Go la rende estremamente semplice e *potrebbe* portare all’esecuzione in parallelo (gestito dal runtime)

### Parallelismo
Riguarda l’esecuzione e permette di fare molte cose contemporaneamente (eseguendo su core diversi)

---
## Goroutine
Le Goroutine sono l’unità fondamentale della concorrenza in Go. Si tratta infatti di funzioni eseguire in modo concorrente, gestite dal runtime di Go.

Sono inoltre molto più economiche dei thread di sistema (migliaia di Goroutine possono coesistere) ed è possibile avviarle anteponendo la parola chiave `go` alla chiamata di funzione.

```go
func main() {
	// goroutine separata (eseguita concorrentemente)
	go func() {
		for i := 0; i<10; i++ {
			fmt.Println("Goroutine:", i)
		}
	}()
	
	// goroutine principale (main)
	for j:=0; j<10; j++ {
		fmt.Println("Main:", j)
	}
}
```

>[!warning]
>Il main NON aspetta la Goroutine secondaria, è per questo necessaria la sincronizzazione (es. un canale o `WaitGroup`)

### Goroutine vs. thread
| Caratteristica           | Goroutine (Go)                                                    | Thread (OS)                                                        |
| ------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| *Gestione*               | gestite dal Runtime di Go (scheduler $M:N$)                       | gestite dal Kernel del Sistema Operativo                           |
| *Dimensioni dello stack* | molto piccolo, tipicamente 2KB (cresce e si riduce dinamicamente) | grande e fisso, tipicamente 1–8MB (dipende dall’OS)                |
| *Costo di creazione*     | estremamente basso (veloce)                                       | relativamente alto (più lento)                                     |
| *Numero tipico*          | decine di migliaia o centinaia di migliaia                        | centinaia o migliaia (il limite è rigido)                          |
| *Commutazione (switch)*  | commutazione più veloce (meno overhead)                           | commutazione più lenta (richiede un cambio di contesto del kernel) |
| *Modello*                | multiplexing $M:N$ (M Goroutine su N Thread del SO)               | modello $1:1$ (un Thread Go = un Thread del SO)                    |

### Lo scheduler (modello $M:N$)
Mentre i thread sono programmati direttamente dal kernel, le Goroutine sono gestire dallo scheduler di Go, che si trova all’interno del runtime:
