---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Closure
Una funzione che fa riferimento a variabili dall’esterno è chiamata **closure** ed ogni sua istanze è legata alla propria copia delle variabili esterne  (nell’esempio un’unica variabile `sum` per ogni chiamata ad `adder()`).

```go
package main
import 'fmt'

func intSeq() func() int {
    i := 0
    return func() int {
        i++
        return i
    }
}

func main() {
    nextInt := intSeq()
    fmt.Println(nextInt()) // 1
    fmt.Println(nextInt()) // 2
    fmt.Println(nextInt()) // 3
	
    newInts := intSeq()
    fmt.Println(newInts()) // 1
}
```