---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## defer
Lo statement `defer` serve a posticipare l’esecuzione di una funzione fino a quando la funzione circostante non ritorna.
Bisogna però fare attenzione, infatti gli argomenti della funzione `defer` vengono valutati **immediatamente** ma la chiamata alla funzione è posticipata

Le istruzione `defer` vengono eseguire nell’ordine LIFO

```go
package main
import "fmt"

func test() {
	defer fmt.Println(" world") // (2) eseguita prima di ritornare
	fmt.Println(" cruel")       // (1) eseguita immediatamente
}

func main() {
	defer fmt.Println("!")      // (3) eseguita per ultima (LIFO)
	fmt.Println("hello")        // (0) eseguita immediatamente
	test()                      // (1) e (2)
}

// out: hello cruel world!
```

Tipicamente viene utilizzato ad esempio per pianificare la chiusura di un file al termine della funzione.