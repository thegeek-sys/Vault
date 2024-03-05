---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduzione
Un campo (detto anche variabile di istanza) costituisce la **memoria privata** di un oggetto (normalmente i campi di una classe sono privati).
Ogni campo ha un **tipo di dati** e un **identificatore** (nome) fornito dal programmatore

---
## Struttura
```java
private [static] [final] tipo_di_dati nome;
```

- `static`
	indica se un campo è **condiviso da tutti**. Definire un campo statico vuol dire che ogni oggetto condividerà la stessa variabile (ci sta solo una locazione di memoria ,se lo modifico in un oggetto verrà modificato in tutti gli oggetti). Un campo di questo tipo è detto *campo di classe*
- `final`
	Indica se il campo è una costante. Questo campo quindi **non può essere modificato**