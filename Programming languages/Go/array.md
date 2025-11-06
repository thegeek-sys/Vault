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

#### Estensione della slice
E’ possibile estendere una slice verso destra (aumentando la sua lunghezza) ed è possibile farlo fino alla sua capacità massima

```go
s := []int{2, 3, 5, 11, 13} // len=6 cap=6
s = s[:0] // s=[], len=0, cap=6
s = s[:4] // s=[2 3 5 7], len=4, cap=6
```

>[!warning]
>- non è possibile estendere una slice verso sinistra (non si può cambiare l’indice di partenza dopo la creazione della slice)
>- non si può indicizzare oltre la `len` se questa è inferiore a `cap`
>- si può indicizzare oltre la `len` se si effettua un re-slicing
>
>>[!example]
>>```go
>>a := []int{0, 1, 2, 3, 4} // len=5, cap=5
>>b := a[1:4] // b=[1 2 3], len=3, cap=4
>>fmt.Println(b[3]) // errore, index 3 fuori dalla lunghezza
>>fmt.Println(b[:cap(b)][3]) // b[:cap(b)]=[1 2 3 4], out=4
>>```

### Funzioni di slice
#### $\verb|make|$
La funzione `make` alloca un array azzerato e restituisce una slice che lo referenzia (viene utilizzato per inizializzare uno slice literal, non possibile altrimenti)

```go
a := make([]int, 5) // a=[0 0 0 0 0], len(a)=5, cap(a)=5
```

Per specificare una capacità diversa dalla lunghezza, si passa un terzo argomento a `make`

```go
b := make([]int, 0, 5) // b=[], len(b)=0, cap(b)=5
```

#### $\verb|append|$

```go
func append(s []T, vs ...T) []T
```

Il primo parametro è una slice di tipo `T`, mentre i parametri successivi sono i parametri di tipo `T` (variable arguments) e restituisce una nuova slice con i nuovi elementi aggiunti

>[!info]
>Alloca un array più grande se necessario (se la capacità è esaurita)

>[!question] Cosa contiene `s`?
>```go
>s := []int{9000}[:0]
>s = append(s, 0)
>```
>
>>[!done] Risposta
>>- `len(s)=1`
>>- `cap(s)=1`
>>- `s=[0]` → infatti inizialmente `s` è `[9000]` con `len=1` e `cap=1`. `s[:0]` lo affetta a `[]` con `len=0` e `cap=1`. `append(s,0)` usa la capacità esistente, impostando l’elemento `0` e restituendo una slice `[0]` con `len=1` e `cap=1`

