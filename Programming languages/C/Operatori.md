---
Created: 
Class: "[[Sistemi Operativi]]"
Programming language: "[[C]]"
Related:
  - "[[Variabili ed espressioni]]"
---
---
## Introduction
Gli operatori sono i seguenti:
- **aritmetici** → $\verb|+, -, *, /, %|$
- **relazionali** → $\verb|==, !=, <, <=, >, >=|$
- **logici** → $\verb|!, &&, ||$
- **bitwise** → $\verb|&, ~, ^, ||$
- **shift** → $\verb|>>, <<|$

---
## Operatori aritmetici
### Regole di precedenza
Le espressioni dentro parentesi tonde `()` vengono valutate per prime. In caso di parentesi innestate si inizia a valutare dalla più esterna

```c
x = a + (a + (3/5 + c * (c - 3)))
```

`*`, `/` o `%` vengono valutate per seconde. Se ce ne sono diverse la valutazione è da sinistra a destra

```c
x = a*b/c
```

Infine `+` o `-` vengono valutate per ultime

### Operatore di divisione $\verb|/|$
Il risultato dipende dai tipi di dato degli operandi. Il risultato è dello stesso tipo di dato dell’operando più grande (in senso lato). Se ad esempio facciamo una divisione tra `float` e `int`, il tipo del risultato sarà `float`

>[!warning]
>Anche se assegni il risultato a un `float` o `double`, se entrambi gli operandi sono `int`, **la divisione sarà comunque intera**:
>```c
int a = 7, b = 2;
float c = a / b;  // c sarà 3.0, perché a/b è una divisione intera!
>```
>
>Per risolvere è necessario fare un cast esplicito a `float` o `double`:
>```c
float c = (float)a/b; // ora c sarà 3.5
>```

### Operazione resto $\verb+\%+$
Ritorna il resto della divisione

### Altre regole
Troncamento
```c
int x;

x=3.14 // x conterrà solo 3, non dà warning
```