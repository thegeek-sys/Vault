---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Index

- [[#Concorrenza vs. parallelismo|Concorrenza vs. parallelismo]]
	- [[#Concorrenza vs. parallelismo#Concorrenza|Concorrenza]]
	- [[#Concorrenza vs. parallelismo#Parallelismo|Parallelismo]]
- [[#Goroutine|Goroutine]]
	- [[#Goroutine#Goroutine vs. thread|Goroutine vs. thread]]
	- [[#Goroutine#Lo scheduler (modello $M:N$)|Lo scheduler (modello $M:N$)]]
- [[#Sincronizzazione|Sincronizzazione]]
	- [[#$ verb sync.WaitGroup $|sync.WaitGroup]]
	- [[#Sincronizzazione#Canali|Canali]]
	- [[#Sincronizzazione#$ verb select $ multiplexing I/O|select - multiplexing I/O]]
		- [[#Gestione del timeout]]
- [[#Sincronizzazione manuale|Sincronizzazione manuale]]
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
1. **M:N scheduling** → lo scheduler di Go multiplexa molte Goroutine ($M$) su un numero molto minore di thread del sistema operativo ($N$)
2. **blocco non fatale** → se una Goroutine si blocca (es. aspettando un I/O di rete), il runtime di Go sposta solo quella singola Goroutine in attesa, non blocca l’intero thread del sistema operativo su cui era in esecuzione. Il thread del So viene immediatamente riassegnato a un’altra Goroutine pronta per l’esecuzione
3. **equità** → la commutazione di contesto delle Goroutine è molto più veloce di quella dei thread del SO e permette a Go di garantire equità e prevenire la starvation in modo efficiente

---
## Sincronizzazione
Per garantire che una o più goroutine completino il loro lavoro prima che il programma principale termini, è necessario un meccanismo di sincronizzazione.

Ci sono due modi principali per sincronizzare le goroutine:
- `sync.WaitGroup` → per attendere il completamento
- canali

### $\verb|sync.WaitGroup|$
Il `WaitGroup` è specificamente progettato per questo scopo: aspettare che un insieme di goroutine abbia completato le proprie operazioni.

Vediamo i principali comandi:
- `wg.Add(N)` → incrementa un contatore per il numero di goroutine da attendere
- `wg.Done()` → decrementa il contatore all’interno della goroutine (tipicamente con `defer`)
- `wg.Wait()` → blocca la goroutine chiamante (`main`) finché il contatore non torna a zero

```go
func main() {
	var wg sync.WaitGroup
	wg.Add(1) // dichiamo al WaitGroup di aspettare 1 goroutine
	
	go func() {
		// garantisce che il contatore sia decrementato, anche in 
		// caso di panic
		defer wg.Done()
		for i:=0; i<5; i++ {
			fmt.Println("Goroutine: ", i)
			time.Sleep(100 * time.Millisecond)
		}
	}()
	
	fmt.Println("Main in attesa...")
	wg.Wait() // main si blocca finché la goroutine non chiama wg.Done()
	fmt.Println("Programma completato")
}
```

### Canali
Se le goroutine devono anche scambiarsi dati, i canali sono lo strumento più idiomatico. Il canale non solo trasferisce dati, ma blocca l’esecuzione finché sia l’invio he la ricezione non sono pronti. 

Se il `main` attende di ricevere un segnale sul canale, attenderà fino a quando la goroutine non lo invierà

```go
func main() {
	// canale vuoto, usato solo come segnale
	done := make(chan bool)
	
	go func() {
		for i:=0; i<5; i++ {
			fmt.Println("Goroutine: ", i)
			time.Sleep(100*time.Millisecond)
		}
		done <- true // invia un segnale al main
	}()
	
	fmt.Println("Main in attesa del segnale...")
	<-done // main si blocca in attesa di un valore dal canale 'done'
	fmt.Println("Programma completato")
}
```

I canali sono dunque il mezzo primario per la comunicazione e la sincronizzazione tra Goroutine. Vediamo ora i principali comandi:
- creazione
	- `make(chan Type)` → canale non bufferizzato
	- `make(chan Type, N)` → canale bufferizzato con capacità $N$
- operazioni (bloccanti)
	- `channel <- value` → invio
	- `v := <-channel`

>[!example] Esempio di sincronizzazione
>I canali non bufferizzati garantiscono che invio e ricezione avvengano simltaneamente (*rendezvous*).
>
>Il seguente codice garantisce che i messaggi siano consumati in ordine
>```go
>fun main() {
>	// canale non bufferizzato: richiede rendezvous
>	var ch = make(chan int)
>	
>	go func() {
>		var i=0
>		for i<10 {
>			ch <- i // 1. invio: si blocca finché qualcuno non riceve
>			i++
>		}
>	}()
>	
>	var j = 0
>	for j < 10 {
>		// 2. ricezione: si blocca finché qualcuno non invia
>		fmt.Println(<-ch)
>		j++
>	}
>}
>```
>
>Il seguente codice invece non garantisce che i messaggi siano consumati in ordine
>```go
>func main() {
>	// creata struttura dati per canale e 'ch' contiene puntatore al canale
>	var ch = make(chan int)
>	
>	// la goroutine 1 riceve una copia del puntatore
>	go senderGoroutine(ch)
>	
>	// la goroutine 2 riceve una copia del puntatore
>	go receiverGoroutine(ch)
>	
>	// ...
>}
>
>func senderGoroutine(ch chan int) {
>	ch <- 42 // invia sulla struttura condivisa
>}
>
>func receiverGoroutine(ch chan int) {
>	v := <-ch // riceve dalla struttura condivisa
>}
>```

### $\verb|select|$: multiplexing I/O
L’istruzione `select` permette a una Goroutine di attendere su più operazioni di canali contemporaneamente. In particolare blocca finché uno dei suoi `case` (operazioni su canali) è pronto (se più canali sono pronti, ne sceglie uno casualmente).
Se è presente un `default`, `select` lo esegue se nessun canale è immediatamente pronto

```go
select {
case v1 := <-chan1:
	fmt.Printf("ricevuto %v dal chan1\n", v1)
case chan2 <- 1:
	fmt.Printf("valore inviato a chan2")
default:
	fmt.Println("nessun canale è pronto")
}
```

#### Gestione del timeout
Un caso d’uso comune di `select` è l’implementazione di timeout usando `time.After`. Infatti `time.After(duration)` restituisce un canale che riceve un valore dopo la durata specificata

Se il canale di timeout è pronto prima del canale dati, l’operazione scade

```go
select {
case v1 := <-dataChannel:
	// operazione riuscita
	fmt.Println("ricevuto dato: %v\n", v1)
case <-time.After(5*time.Second):
	// operazione fallita per timeout
	fmt.Println("timeout: nessun dato ricevuto")
}
```

---
## Sincronizzazione manuale
Quando è necessario condividere la memoria (variabili) invece di comunicare tramite canali, si usano i mutex (`sync.Mutex`) per evitare race condition. Un mutex infatti ha il compito di proteggere una sezione critica del codice e solo un Goroutine alla volta può ottenere il lock

Operazioni:
- `mu.Lock()` → acquisisce il lock; blocca se già detenuto
- `mu.Unlock()` → rilascia il lock

```go
var counter int
var mu sync.Mutex // mutex per proteggere counter

// Increment() è sicura per l'uso concorrente
func Increment() {
	mu.Lock() // inizio sezione cirica
	defer mu.Unlock() // garantisce rilascio lock, anche in caso di panic
	counter++ // accesso sicuro alla risorsa condivisa
}
```