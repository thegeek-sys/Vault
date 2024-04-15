---
Created: 2024-04-10
Programming language: "[[Java]]"
Related:
  - "[[Interfacce]]"
Completed:
---
---
## Introduction
E' possibile definire **classi anonime** (ovvero senza nome) che implementano un'interfaccia o estendono una classe
Queste sono utilizzate per **creare un’unica istanza** (utili ad esempio per creare un iteratore al volo)

Sintassi:
```java
TipoDaEstendere unicoRiferimentoAOggetto = new TipoDaEstendere() {
	// codice della classe anonima (implementazione dall'interfaccia)
	// o estensione della classe
};
```

### Esempio
Consideriamo la seguente interfaccia
```java
public interface Formula {
	double calculate(int a);
	default double sqrt(int a) { return Math.sqrt(a); }
}
```

Adesso creiamo una classe anonima che implementa l’interfaccia
```java
Formula formula = new Formula() {
	@Override
	public double calculate(int a) {
		return Math.sqrt(a * 100);
	}
}

formula.calculate(100); // 100.0
formula.sqrt(16); // 4.0
```

---
## Interfacce funzionali
In Java 8 è disponibile una nuova annotazione `@FunctionalInterface` direttiva di programmazione che controlla se quella che segue è effettivamente una functional interface ovvero garantisce che l’interfaccia sia dotata esattamente di **un solo metodo astratto**

```java
@FunctionalInterface
public interface Runnable {
	void run();
}
```

---
## Espressioni lambda
In Java 8 è possibile specificare funzioni utilizzando una notazione molto compatta, le espressioni lambda:

```java
() -> { System.out.println("hello, lambda!"); }

(tipo1 nome_p1, ..., tipon nome_pn) -> { codice della funzione }
```

Tali espressioni creano oggetti anonimi assegnabili a riferimenti a interfacce funzionali compatibili con l’intestazione (intput/output) della funzione creata

```java
Runnable r = () -> { System.out.println(ʺhello, lambda!ʺ); };
r.run(); // "hello, lambda!"
```

Il tipo dei parametri è **opzionale** perché si ricava dal contesto dell’interfaccia a cui facciamo riferimento
Le parentesi tonde sono **opzionali** se in input abbiamo un solo parametro
Le parentesi graffe sono **opzionali** se il codice è costituito da una sola riga
**Non è necessario nessun return** se il codice è dato dall’espressione di ritorno

```java
(int a, int b) -> { return a+b; }
(a, b) -> { return a+b; }
(a, b) -> return a+b;
(a, b) -> a+b;

(String s) -> { return s.replace(" ", "_"); }
(s) -> { return s.replace(" ", "_"); }
s -> { return s.replace(" ", "_"); }
s -> return s.replace(" ", "_");
s -> s.replace(" ", "_");
```

Dunque ritornando al nostro [[#Introduction#Esempio|esempio]] da Java 8 può essere modificato in:
```java
Formula formula1 = a -> sqrt(a * 100);
Formula formula2 = a -> a*a;
Formula formula3 = a -> a-1
```

### Esempio: conversione da un tipo F a un tipo T
```java
@FunctionalInterface
public interface Convert<F,T> {
	T convert(F from);
}

Converter<String,Integer> converter = from -> Integer.valueOf(from);
Integer converted = converter.convert("123")
Converter<String, MyString> stringConverter = a -> new MyString(a);
MyString myString = stringConverter.convert(“123”);
```

---
## Single Abstract Method (SAM) type
Le interfacce funzionali sono di tipo SAM, a ogni metodo che accetta un’interfaccia SAM, si può passare un’espressione lambda compatibile con l’unico metodo dell’interfaccia SAM.
Analogamente per un riferimento a un’interfaccia SAM

---
## Differenza tra class anonime ed espressioni lambda
La parola chive *this*:
 - **classi anonime** → si riferisce all’oggetto anonimo
 - **espressioni lambda** → si riferisce all’oggetto della classe che lo racchiude

La compilazione è differente:
- **classi anonime** → compilate come classi interne
- **espressioni lambda** → compilate come metodi privati invocati dinamicamente

---
## Quando utilizzare le lambda?
E' da consigliare l'impiego delle espressioni lambda principalmente quando il codice si scrive su una sola riga
In alternativa, si preferisce un'implementazione mediante classe o classe anonima (o, vedremo più avanti, riferimenti a metodi)