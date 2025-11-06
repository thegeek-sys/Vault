---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
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