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
## Metodi statici
I metodi statici sono metodi di classe. Questi NON hanno accesso ai campi di istanza ma solo ai campi di classe (ma da un metodo non statico posso accedere ad un campo static)

```java
public class ContaIstanze {
	static private int numberOfInstances;
	
	public ContaIstanze() {
		numberOfInstances++;
	}
	
	public static void main(String[] args) {
		new ContaIstanze();
		new ContaIstanze();
		new ContaIstanze();
		
		// accesso da metodo statico a campo statico
		System.out.println("Istanze: "+numberOfInstances);
	}
}
```

---
## Esempi

```java
public int getValue() { return value; }
public void reset(int newValue) { value = newValue; }
```