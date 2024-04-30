---
Created: 2024-04-30
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Un tipo generico permette di generalizzare e rendere ancora più riutilizzabile il codice.
I tipi generici, in Java, sono un modello di programmazione che permette di **definire**, con una sola dichiarazione, **un intero insieme di metodi o di classi**. Risultano quindi essere uno strumento molto potente ma da usare con cautela.

>[!hint] I generici funzionano solo con i tipi derivati
>Non è possibile utilizzare i tipi primitivi, ad es. `int`, `double`, `char`, ecc.

### Esempio
```java
public class Valore<T> {
	private final T val;
	
	public Valore(T val) { this.val = val; }
	public T get() { return val; }
	
	@Override
	public String toString() { return ""+val; }
	
	public String getType() { return val.getClass().getName(); }
}
```

Per definire un tipo generico della classe, si utilizza la sintassi a **parentesi angolari** dopo il nome della classe con il tipo generico da utilizzare. Da quel punto, si utilizza il tipo generico come un qualsiasi altro tipo di classe.
#### Istanziare una classe generica
```java
public static void main(String[] args) {
	Valore<Integer> i = new Valore<>(42);
	Valore<String> s = new Valore<>("ciao");
	Valore<Valore<String>> v = new Valore<>(s);
	
	System.out.println(i.get()+": "+i.getType());
	System.out.println(s.get()+": "+s.getType());
	System.out.println(v.get()+": "+v.getType());
}

/* OUTPUT */
// 42: java.lang.Integer
// ciao: java.lang.String
// ciao: Valore
```

### Esempio
```java
public class Coppia<T> {
	private T a,b;
	
	public Coppia(T a, T b) {
		this.a = a;
		this.b = b;
	}
	public T getPrimo() { return a; }
	public T getSecondo() { return b; }
}


Coppia<Integer> ci = new Coppia<Integer>(10, 20);
Coppia<Double> cd = new Coppia<Double>(10.5, 20.5);
Coppia<String> cs = new Coppia<String>("abc", "def");
Coppia<Object> co = new Coppia<Object>("abc", 20);
```

---
## Specificare più tipi generici di classe
In questo caso i tipi generici sono separati da una virgola. Per convenzione i tipi generici son chiamati con le lettere **T**, **S**, ecc. (**E** nel caso in cui siano elementi di una collection)

```java
public class Coppia<T, S> {
	private T a;
	private S b;
	
	public Coppia(T a, S b) {
		this.a = a;
		this.b = b;
	}
	public T getPrimo() { return a; }
	public S getSecondo() { return b; }
}
```

---
## Estendere le classi generiche
Ovviamente è possibile estendere le classi generiche per creare classi più specifiche

Ad esempio, una classe `Orario` può estendere la classe `Coppia`:
```java
public class Orario extends Coppia<Integer, Integer> {
	public Orario(Integer a, Integer b) {
		super(a, b);
	}
}
```

O una classe `Data`:
```java
public class Data extends Coppia<Integer, Coppia<Integer, Integer>> {
	public Data(Integer giorno, Integer mese, Integer anno) {
		super(giorno, new Coppia<Integer, Integer>(mese, anno));
	}
}
```