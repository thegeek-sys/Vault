---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Valori funzione
Le funzioni possono essere passare come qualsiasi altro valore alle funzioni

```go
func hypot(x, y, float64) float64 {
	return math.Sqrt(x*x + y*y)
}
func compute(fn func(float64, float64) float64) float64 {
	return fn(3,4)
}

func main() {
	fmt.Println(compute(hypot)) // 5
	fmt.Println(compute(math.Pow)) // 81 (3^4)
}
```