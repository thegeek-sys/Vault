---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## switch
Lo `switch` consiste in una sequenza di istruzioni `if - else` ottimizzata. I `case` dello switch in Go vengono valutati dall’alto al basso (viene eseguito il primo `case` la cui espressione corrisponde alla condizione dello `switch`)

>[!warning]
>- i valori dei `case` non devono essere costanti
>- i valori non devono essere per forza interi

```go
package main
import (
	"fmt"
	"runtime"
)

func main() {
	switch os := runtime.GOOS; os {
	case "darwin":
		fmt.Println("macOS.")
	case "linux":
		fmt.Println("Linux.")
	default:
		// freebsd, openbsd,
		// plan9, windows...
		fmt.Printf("%s.\n", os)
	}
}
```