---
Created: 2024-04-08
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [Introduction](#introduction)
- [Metodi di default e statici](#metodi-di-default-e-statici)
	- [Metodi privati](#metodi-privati)
- [Dichiarazione di un’interfaccia](#dichiarazione-di-uninterfaccia)
	- [Implementare u’interfaccia](#implementare-uinterfaccia)
- [Contratto](#contratto)
- [Esempio: iterabile](#esempio-iterabile)
- [Iterable e Iterator](#iterable-e-iterator)
	- [Esempio](#esempio)

---
## Introduction
Le interfacce sono uno strumento che Java mette a disposizione per consentire a più classi di fornire e implementare un insieme di metodi comuni
Le interfacce definiscono e standardizzano l’interazione fra oggetti tramite un insieme limitato di operazioni.
Esse specificano soltanto il comportamento (le classi astratte possono definire anche un costruttore) che un certo oggetto deve presentare all’esterno, cioè cosa quell’oggetto può fare. L’implementazione di tali operazioni, cioè come queste vengono tradotte e realizzate, rimane invece non definito

---
## Metodi di default e statici
E' possibile specificare delle implementazioni di default di metodi non statici. Questo viene fatto attraverso la parola chiave `default`. Ciò è dovuto nelle interfacce dall’estensione di interfacce rilasciate in precedenza con metodi senza rompere il contratto con il codice che utilizza le versioni precedenti

E’ inoltre importante ricordare che i metodi statici **non godono del polimorfismo** in quanto sono metodi di utilità non associati alle singole istanze

### Metodi privati
Per facilitare il riuso del codice, da Java 9 è possibile definire metodi privati all’interno di un’interfaccia (questi possono essere chiamati solamente dall’interfaccia stessa)

```java
public interface Loggabile {
	private void log(String msg, LogLevel level) {
		// codice per il logging
	}
	
	default void logInfo(String msg) { log(msg, LogLevel.INFO); }
	default void logWarn(String msg) { log(msg, LogLevel.WARNING); }
	default void logError(String msg) { log(msg, LogLevel.ERROR); }
	default void logFatal(String msg) { log(msg, LogLevel.FATAL); }
}
```

---
## Dichiarazione di un’interfaccia
Un’interfaccia è una classe che può contenere soltanto:
- costanti
- metodi astratti
- metodi default e metodi statici (Java 8)
- metodi privati, tipicamente da invocare in metodi di default (Java 9)

Tutti i **metodi** dichiarati in un’interfaccia sono implicitamente `public abstract`
Tutti i **campi** dichiarati in un’interfaccia sono implicitamente `public static final`
Tranne nel caso dei metodi di default o statici, non è possibile specificare alcun dettaglio implementativo (non vi è alcun corpo di metodo o variabile di istanza)

```java
public interface SupportoRiscrivibile {
	// implicitamente final static public
	int TIMES = 1000;
	
	// implicitamente public abstract
	void leggi();
	void scrivi();
}
```

### Implementare u’interfaccia
Per realizzare un’interfaccia è necessario che una classe la implementi tramite la parola chiave `implements`
Una classe che implementa una interfaccia decide di voler **esporre** pubblicamente all’esterno il comportamento descritto dall’interfaccia
E’ obbligatorio che ciascun abbia esattamente la **stessa intestazione** che esso presenta nell’interfaccia

```java
public class Nastro implements SupportoRiscrivibile {
	private Pellicola pellicola;
	
	@Override
	public void leggi() {
		attivaTestina();
		muoviTestina();
	}
	
	@Override
	public void scrivi() {
		attivaTestina();
		caricaTestina();
		muoviTestina();
		scaricaTestina();
	}
	
	public void attivaTestina() {}
	public void caricaTestina() {}
	public void scaricaTestina() {}
	public void muoviTestina() {}
}
```

Anche la classe `MemoriaUsb` implementa l’interfaccia `SupportoRiscrivibile`, definendo i propri metodi `leggi()` e `scrivi()`
```java
public class MemoriaUsb implements SupportoRiscrivibile {
	private CellaMemoria[] celle;
	
	@Override
	public void leggi() {
		// leggi la cella corretta
	}
	
	@Override
	public void scrivi() {
		// modifica la cella corretta
	}
}
```

> [!hint]
> Le interfacce permettono di modellare comportamenti comuni a classi che non sono necessariamente in relazione gerarchica (is-a, è-un).
> Nel momento in cui una classe C decide di implementare un’interfaccia I, tra queste due classi si instaura una relazione di tipo is-a, ovvero C è di tipo I (comportamento simile all’ereditarietà) quindi anche per le intefacce valgono le regole del polimorfismo
> `SupportoRiscribile supporto = new Nastr();`

---
## Contratto

Implementare un’interfaccia equivale a **firmare un contratto con il compilatore** che stabilisce l’impegno ad implementare tutti i metodi specificati dall’interfaccia o a dichiarare la classe abstract

Ci sono 3 possibilità per una classe che implementa un’interfaccia:
1. fornire un’implementazione concreta di tutti i metodi, definendone il corpo
2. fornire un’implementazione concreata per un sottoinsieme proprio dei metodi dell’interfaccia
3. decidere di non fornire alcuna implementazione concreta
N.B. Negli ultimi due casi, però, la classe va dichiarata abstract

>[!faq] Se implementando un’interfaccia devo dichiarare tutti i metodi in essa definiti, perché non ricorrere ad una classe astratta?
> Poiché potrebbe essere necessario estendere più di una classe, ma in Java ciò non è possibile in quanto `extends` può essere seguito solo da un unico nome di classe. Al contrario una calsse può implementare tutte le interfacce desiderate
> > [!info]- UML
> > ![[UML#Interfacce]]

> [!faq] Cosa succede se due metodi di default vengono ereditati da due interfacce implementate?
> E' necessario implementare quel metodo nella classe che implementa le due interfacce e "disambiguare" il metodo chiamando quello/i appropriato/i con la sintassi
> `Interfaccia.super.metodo(...);`

---
## Esempio: iterabile
Ci sono molte classi di natura diversa che rappresentano sequenze di elementi, tuttavia le sequenze hanno qualcosa in comune: è possibile iterare sui loro elementi

```java
public interface Iterabile {
	boolean hasNext();
	Object next();
	void reset();
}
```

Ciascuna classe implementerà i metodi a suo modo:
```java
public class MyIntegerArray implements Iterabile {
	private Integer[] array;
	private int k = 0;
	
	public MyIntegerArray(Integer[] array) {
		this.array = array;
	}
	
	@Override
	public boolean hasNext() { return k < array.length; }
	
	@Override
	public Object next() { return array[k++]; }
	
	@Override
	public void reset() { k=0; }
}


public class MyString implements Iterabile {
	private String s;
	private int k = 0;
	
	public MyString(String s) {
		this.s = s;
	}
	
	@Override
	public boolean hasNext() { return k < array.length; }
	
	@Override
	public Object next() { return s.charAt(k++); }
	
	@Override
	public void reset() { k=0; }
}
```

Il problema di queste due implementazioni sta nel fatto che **non ci permette di avere iteratori multipli** (per esempio non posso fare due for nestati che ciclino sullo stesso iterabile con due iteratori diversi). A soluzione di questo problema Java ci mette a disposizione le interfacce `Iterable` e `Iterator`

---
## Iterable e Iterator
Queste due interfacce standard di Java ci permettono di disaccoppiare l'oggetto su cui iterare dall'oggetto che tiene la posizione d'iterazione. Infatti senza di essi, utilizzando due for nestati, entrambi i for avranno lo stesso puntatore sull’iterbile (aumentando il primo aumenta anche il secondo)


**`java.lang.Iterable`**

| Modifier and Type | Method and Description                                                                |
| ----------------- | ------------------------------------------------------------------------------------- |
| `Iterator<T>`     | `iterator()`<br>Restituisce ogni volta che lo chiamo una nuova istanza dell’iteratore |


**`java.lang.Iterator`**
Questa è fondamentale in quanto permette di **iterare su collezioni**. E’ in relazione con l’interfaccia Iterable nel senso che chi implementa Iterable restituisce un Iterator sull’oggetto-collezione

| Modifier and Type | Method and Description                                        |
| ----------------- | ------------------------------------------------------------- |
| `boolean`         | `hasNext()`<br>Ritorna true se l’iterazione ha altri elementi |
| `E`               | `next()`<br>Ritorna il prossimo elemento dell’iterazione      |
| `void`            | `remove()`<br>                                                |

### Esempio
```java
import java.util.ArrayList;
import java.util.Iterator

// rappresenta un Jukebox di canzoni su cui si può iterare
public class Jukebox implements  Iterable<Canzone> {
	// elenco di canzoni
	private ArrayList<Canzone> canzoni = new ArrayList<Canzone>();
	
	// permette di aggiungere una canzone
	public void addCanzone(Canzone c) {
		canzoni.add(c);
	}
	
	@Override
	public Iterator<Canzone> iterator() {
		return canzoni.iterator();
	}
}
```