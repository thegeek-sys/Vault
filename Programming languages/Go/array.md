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