---
Created: 2024-02-29
Programming language: "[[Java]]"
Related:
  - "[[Primitivi]]"
Completed:
---
---
## Index
- [[#Incrementi|Incrementi]]
- [[#Operatori booleani|Operatori booleani]]
- [[#Relazionali]]
- [[#Operatore ternario]]
- [[#Shift]]
---

## Incrementi
- Post-incremento:
	- `var++` (var = var +1)
	- `var--`
- Pre-incremento
	- `++var`
	- `--var`
 
*pre vs post-incremento*:
In un’espressione il post-incremento a++, prima esegue l’operazione con a, e poi lo incrementa di 1. Dunque:

```java
int a = 3;
int c = a++ // c == 3, a == 4

int a = 3;
int c = ++a // c == 4, a == 4

int a = 4;
int c = 3;
int z = (a++) - (c--); // z == 1, a == 5, c == 2
```

---
## Operatori booleani
- && → and logico 
- || → or
- ! → not
- ^ → xor
&  e | - and  e or bit a bit (per i binari)

---
## Relazionali
- ==
- !=
- < , <= , > , >=
- istanceof

---
## Operatore ternario
- ? :

---
## Shift
- <<,  >>, >>>
utili per i numeri binari: ogni shift a sinistra moltiplica per 2 (aggiungo uno 0 a destra in un numero binario)
