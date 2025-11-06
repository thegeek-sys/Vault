---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## array
`[n]T` è un array di `n` valori di tipo `T`. Gli array non possono essere ridimensionati, la dimensione è fissa

```go
var a [10]int // dichiarazione
primes := [6]int{2, 3, 5, 7, 11, 13}
```

>[!warning] La lunghezza è parte del tipo!
>Un `[6]int` è un tipo diverso da un `[5]int`

>[!question] Qual è il contenuto del seguente codice?
>```go
>v := [6]int{2, 3, 5, 7}
>```
>>[!done] Risposta
>>```go
>>[2 3 5 7 0 0]
>>```
>>
>>Infatti i valori non specificati vengono inizializzati al valore di default del tipo

---
## Slice
Uno slice è una vista a dimensione dinamica su un array sottostante

```go
// low incluso, high escluso
a[low : high]
```

E ha come tipo `[]T` (non specifica la dimensione) e non serve a memorizzare dati direttamente, ma fa riferimento a una sezione di un array.
La modifica degli elementi di una slice modifica l’array sottostante e qualsiasi slice corrispondente che punta allo stesso array.

### Slice literals
Anche gli slice literals hanno come valore di default `nil`, ma a differenza degli array sono dinamici ed è quindi possibile utilizzare delle funzioni per poterne modificare la dimensione

```go
[]bool{true, true, false}
```

### Lunghezza e capacità della slice
- `len` → restituisce il numero di elementi contenuti nella slice
- `cap` → restituisce il numero di elementi nell’array sottostante, a partire del primo elemento della slice

```go
s := []int{1, 2, 3, 4, 5}
fmt.Println(len(s))  // 5
fmt.Println(cap(s))  // 5

sub := s[1:3]
fmt.Println(sub)       // [2 3]
fmt.Println(len(sub))  // 2 (da indice 1 a 2)
fmt.Println(cap(sub))  // 4 ([1, 2, 3, 4])
```