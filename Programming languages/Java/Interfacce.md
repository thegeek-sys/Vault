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