---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
  - "[[Programming languages/Go/array|array]]"
---
---
## for
In Go esiste un solo tipo di loop: il `for`. Nella sintassi non si ha nessuna parentesi `()` dopo il `for` statement ma sono sempre necessarie le parentesi graffe `{}`

```go
sum := 0
for i:=0: i<10; i++ {
	sum += i
}
```

>[!tip]
>Può anche essere scritto in questa maniera
>```go
>sum := 0
>i := 0 // init statement
>for ; ; {
>	if i>=10 {break} // condition
>	sum += i
>	i++ // post statement
>}
>```

---
## while
Visto che `init` e `post` sono opzionali, è possibile utilizzare il `for` anche come un while

```go
sum := 1
for sum<1000 {
	sum += sum
}
```

---
## $\verb|range|$
Una forma del ciclo `for` che itera su una slice, un array, una stringa o una mappa è `range`. Questa funzione restituisce due valori per iterazione: indice e copia dell’elemento (o chiave e valore per le mappe)

```go
// pow nome slice
for i, v := range pow {
	// i è l'indice, v è la copia dell'elemento
}

// per usare solo l'indice e ignorare il valore (obbligatorio _)
for i, _ := range pow {
	// ...
}

// forma abbreviata per usare solo l'indice
for i := range pow {
	// ...
}

// per usare solo il valore e ignorare l'indice (obbligatorio _)
for _, v := range pow {
	// ...
}
```