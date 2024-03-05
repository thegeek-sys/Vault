---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

- [[#Introduzione|Introduzione]]
- [[#Struttura|Struttura]]
- [[#Esempi|Esempi]]
---
## Introduzione
Un metodo è tipicamente **pubblico**, ovvero visibile a tutti. Il nome di un metodo per convenzione inizia con una lettera minuscola, mentre le parole seguenti iniziano con lettera minuscola (es. dimmiTuttoQuelloCheSai() ).

---
## Struttura
```java
public tipo_di_dati nomeDelMetodo(tipo_di_dati nomeParam1, ...) {
	istruzione 1;
	.
	.
	.
	istruzione m
}
```

- `tipo_di_dati`
	Indica il tipo di dato in output. Questo può essere `void` che indica che il metodo non restituisce alcun tipo di valore
- `nomeDelMetodo`
	Indica l’identificatore del metodo che deve rispettare la convenzione *CamelCase*
- `(tipo_di_dati nomeParam1, ...)`
	Elenco (eventualmente vuoi) dei nomi dei parametri con relativo tipo

---
## Esempi

```java
public int getValue() { return value; }
public void reset(int newValue) { value = newValue; }
```