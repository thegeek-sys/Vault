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

### Per le classi generiche non vale l’ereditarietà dei tipi generici
Ad esempio, `ArrayList<Integer>` non è il tipo di `ArrayList<Number>` o `ArrayList<Object>`:
```java
/* NON LO POSSO FARE */
ArrayList<Number> listaDiNumeri = new ArrayList<Integer>();
```

Ma rimane conunque l’ereditarietà tra classi:
```java
List<Integer> listaDiNumeri = new ArrayList<Integer>();
```

---
## Vincoli sul tipo generico
### Tramite extends
Nonostante si tratti di tipi generici è comunque possibile impostare un vincolo sul tipo che il generico può ricevere attraverso la sintassi

Si può imporre un vincolo sul tipo generico T mediante la parola chiave:
- `extends` → T deve essere un sottotipo della classe specificata o la
classe stessa (covarianza)
- `super` → T deve essere una superclasse della classe specificata o la classe stessa (controvarianza)

```java
<T extends InterfacciaOClasse>
```

In questo modo posso che `T` deve necessariamente essere un sotto tipo di `Classe` o implementare `Interfaccia`

Nell’esempio seguente definisco l’interfaccia `MinMax` e dico che i tipi che riceve in input devono necessariamente implementare l’interfaccia `Comparable`
```java
// utilizzo extends per convenzione nonostante
// Comparable sia un'interfaccia (dovrei usare implements)
// il costrutto significa che T deve essere una classe che
// implementa Comparable<T> (T deve essere comparabile)
public interface MinMax<T extends Comparable<T>> {
	T min();
	T max();
}


public class MyClass<T extends Comparable<T>>
					   implements MinMax<T> {
	// ...
}

/* ERRORI */
class MyClass<T extends Comparable<T>>
				implements MinMax<T extends Comparable<T>> {}
class MyClass implements MinMax<T> {}

/* CORRETTO */
class MyClass implements MinMax<Integer> {}
```

---
## Definire un metodo generico
Per definire un metodo generico con proprio tipo generico è necessario **anteporre il tipo generico tra parentesi angolari al tipo di ritorno**:
```java
// scrivo un metodo che fa riferimento ad un ArrayList ma non
// specifica il tipo degli elementi di ArrayList
public static <T> void (ArrayList<T> lista) {
	for (T o : lista) {
		System.out.println(o.toString());
	}
}
```

In questo caso però, a differenza delle classi generiche, con i metodi generici posso accettare solo il tipo stesso dichiarato dentro le parentesi angolate (NON una sua sottoclasse)
```java
public static void esamina(ArrayList<Frutto> frutti) {
	// accetta solamente ArrayList<Frutto>
	// e non ArrayList<Arancia>
}


public static <T extends Frutto> void esamina(ArrayList<T> frutti) {
	// in questo caso posso prendere in input
	// qualsiasi ArrayList i cui elementi siano di tipo
	// Frutto o qualsiasi suo sottotipo
}
```

---
## Jolly come tipi generici
Nel caso in cui non sia necessario utilizzare il tipo generico T nel corpo della classe o del metodo, è possibile utilizzare il **jolly `?`** (anche detto **wildcard**)
```java
public static void mangia(ArrayList<? extends Mangiabile> frutta) {
	// qui NON posso usare ?
}
```

Equivale a:
```java
public static <T extends Mangiabile> void mangia2(ArrayList<T> frutta) {
	// qui posso usare T
}
```

>[!warning]
>Nel caso in cui utilizzo il simbolo jolly `?` non posso utilizzare il riferimento a quel tipo

Il jolly viene utilizzato nel caso in cui **non sia necessario conoscere il tipo parametrico**, si può utilizzare `<?>`

Ad esempio:
```java
public class Punto<T extends Number> {
	private T x;
	private T y;
	
	public Punto(T x, T y) { this.x = x; this.y = y; }
	
	@Override
	public String toString() { return "("+x+";"+y+")"; }
	
	public static void main(String[] args) {
		Punto<?> p = new Punto<Integer>(10, 42);
		System.out.println(p);
		
		p = new Punto<Double>(11.0, 43.5);
		System.out.println(p);
	}
}
```

### Esempio: metodo generico di somma
Implementare un metodo generico `somma` che calcoli la somma di tutti i numeri contenuti in una collezione
```java
public class SommaNumeriGenerico {
	public static void main(String[] args) {
		// interi
		Integer[] numeri = { 1, 2, 3, 4 };
		ArrayList<Integer> listaDiNumeri = new ArrayList<Integer>(Arrays.asList(numeri));
		
		System.out.printf("La lista contiene: %s\n", listaDiNumeri);
		System.out.printf("La somma dei numeri è: %.lf\n", somma(listaDiNumeri));
	}
	
	public static double somma(ArrayList<? extends Number> lista) {
		double tot = 0.0;
		
		for (Number n : lista) {
			tot += n.doubleValue();
		}
		
		return tot;
	}
}
```

---
## Come funziona dietro le quinte?
I tipi generici in Java sono introdotti attraverso la **cancellazione del tipo** (type erasure).

Infatti quando il compilatore traduce il metodo/la classe generica in bytecode Java:
1. **elimina la sezione del tipo parametrico** e sostituisce il tipo parametrico con quello reale
2. per default **il tipi generico viene sostituito** con il tipo `Object` (a meno di vincoli sul tipo)

>[!hint]
>Solo una copia del metodo o della classe viene creata!

### Esempio
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
Viene trasformato in:
```java
public class Coppia {
	private Object a;
	private Object b;
	
	public Coppia(Object a, Object b) {
		this.a = a;
		this.b = b;
	}
	public Object getPrimo() { return a; }
	public Object getSecondo() { return b; }
}
```

```java
public class Massimo {
	public static <T extends Comparable<T>> T getMassimo(T a, T b, T c) {
		if (a.compareTo(b)>0) return a.compareTo(c) >= 0 ? a : c;
		else return b.compareTo(c) >= 0 ? b : c;
	}
	
	public static void main(String[] args) {
		int max = getMassimo(10, 20, 30);
		String s = getMassimo("abc", "def", "ghi");
	}
}
```
Viene trasformato in:
```java
public class Massimo {
	public static Comparable getMassimo(Comparable a, Comparable b, Comparable c) {
		if (a.compareTo(b)>0) return a.compareTo(c) >= 0 ? a : c;
		else return b.compareTo(c) >= 0 ? b : c;
	}
	
	public static void main(String[] args) {
		int max = (Integer)getMassimo(10, 20, 30);
		String s = (String)getMassimo("abc", "def", "ghi");
	}
}
```

---
## Come ottenere informazioni sull’istanza di un generico?
Per via della cancellazione del tipo, non possiamo conoscere il tipo generico a tempo di esecuzione.
```java
public static void main(String[] args) {
	List<Integer> x = Arrays.asList(4, 2);
	boolean b = x instanceof List<Integer> // ERRORE
}
```

Ma possiamo comunque verificarne il tipo usando il wildcard?
```java
public static void main(String[] args) {
	List<Integer> x = Arrays.asList(4, 2);
	boolean b = x instanceof List<?> // true
}
```