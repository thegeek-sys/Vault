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

// adder restituise una closure
func adder() func(int) int {
    sum := 0 // questa variabile è "catturata" dalla closure
    return func(x int) int {
        sum += x
        return sum
    }
}

func main() {
    pos, neg := adder(), adder()
    for i:=0; i<10; i++ {
	    fmt.Println(
		    // pos e  neg mantengono stati sperati
		    pos(i),
		    neg(-2*i)
	    )
    }
}
```