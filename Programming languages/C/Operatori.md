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
- **aritmetici** → $\verb|+, -, *, /, \%|$
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
Il risultato dipende dai tipi di dato degli operandi. Il risultato è dello stesso tipo di dato dell’operando più grande (in senso lato)