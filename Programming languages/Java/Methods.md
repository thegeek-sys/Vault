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
- [[#Metodi statici]]
- [[#Esempi|Esempi]]
---
## Introduzione
Il miglior modo per sviluppare e mantenere un programma grande è di costruirlo da pezzi piccoli e semplici (principio del *divide et impera*). I metodi permettono di modularizzare un programma separandone i compiti in unità autocontenute.

Le istruzioni di un metodo non sono visibili da un altro metodo (ma possono essere riutilizzate in molti punti del programma). Tuttavia certi metodi non utilizzano lo stato dell’oggetto, ma si applicano all’intera classe (statici).
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
I metodi statici sono metodi di classe. Questi NON hanno accesso ai campi di istanza ma solo ai campi di classe (ma da un metodo non statico posso accedere ad un campo static).

Posso accedere ad essi:
- **dall’interno** semplicemente chiamando il metodo
- **dall’esterno** `NomeClasse.nomeMetodo()` (non devo quindi istanziare alcun nuovo metodo nella heap)

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

### Perché il metodo main() è dichiarato static?
La Java Virtual Machine invoca il metodo `main` della classe specificata ancora prima di aver creato qualsiasi oggetto.
La classe potrebbe non avere un costruttore senza parametri con cui creare l’oggetto

---
## Esempi

```java
public int getValue() { return value; }
public void reset(int newValue) { value = newValue; }
```