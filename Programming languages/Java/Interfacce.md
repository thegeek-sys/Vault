---
Created: 2024-04-08
Programming language: "[[Java]]"
Related: 
Completed:
---
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
> Le interfacce permettono di modellare comportamenti comuni a classi che non sono necessariamente in relazione gerarchica (is-a, è-un)

## Esempio: iterabile
Ci sono molte classi di natura diversa che rappresentano sequenze di elementi, tuttavia le sequenze hanno qualcosa in comune: è possibile iterare sui loro elementi

```java
public interface Iterabile {
	boolean hasNext();
	Object next();
	void reset();
}
```

