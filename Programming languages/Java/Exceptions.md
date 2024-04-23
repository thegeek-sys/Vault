---
Created: 2024-04-23
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Le eccezioni rappresentano un meccanismo utile a notificare e gestire gli errori e vengono generati quando durante l’esecuzione si è verificato un **errore**
Il termine “eccezione” indica un **comportamento anomalo**, che si discosta dalla normale esecuzione e impararle a gestire rende il codice più robusto e sicuro

Questa viene generata ad esempio quando provo ad accedere ad un elemento il cui indice non è presente in un array
```java
int[] estrazioneLotto = { 3, 29, 10, 23, 67 };
for (int i=0; i<=5; i++) System.out.println(estrazioneLotto[i]);

/*
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: 5
*/
```
In questo caso l’esecuzione viene interrotta e ci accorgiamo del superamento incontrollato dei confini dell’array

### Vantaggi
- In linguaggi come il C, la logica del programma e la logica di gestione degli errori sono **interlacciate**: questo rende più difficile leggere, modificare e mantenere il codice
- Gli errori vengono **propagati verso l’alto** lungo lo stack di chiamate
- Codice **più robusto**: non dobbiamo controllare esaustivamente tutti i possibili tipi di errore: il polimorfismo lo fa per noi, scegliendo l’intervento più opportuno
### Svantaggi
- L’**onere** di gestire i vari tipi di errore si sposta sulla JVM che si incarica di capire il modo più opportuno per gestire la situazione di errore

### Cosa si può gestire con le eccezioni
E’ possibile gestire
- Eventi **sincroni**, che si verificano a seguito dell’esecuzione di un’istruzione
	- Errori *non critici* → errori che derivano da condizioni anomali
		- divisione per zero
		- errore di I/O
		- errore durante il parsing
	- Errori *critici* o irrecuperabili → errori interni alla JVM
		- conversione di un tipo non consentito
		- accesso ad una variabile di riferimento con valore `null`
		- mancanza di memoria libera
		- riferimento ad una classe inesistente

**NON** è possibile gestire
- Eventi **asincroni**, che accadono parallelamente all’esecuzione e quindi indipendenti dal flusso di controllo
	- completamenti nel trasferimento I/O
	- recezioni messaggi su rete
	- click del mouse

---
## Eccezioni notevoli

| Eccezione                    | Descrizione                                                                                                        |
| :--------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `IndexOutOfBoundsException`  | Accesso ad una posizione non valida di un array o una stringa (<0 o maggiore della sua dimensione)                 |
| `ClassCastException`         | Cast illecito di un oggetto ad una sottoclasse a cui non appartiene<br>Es. `Object x = new Integer(0); (Stringa)x` |
| `ArithmeticException`        | Condizione aritmetica non valida (es. divisione per zero)                                                          |
| `CloneNotSupportedException` | Metodo `clone()` non implementato o errore durante la copia dell’oggetto                                           |
| `ParseException`             | Errore inaspettato durante il parsing                                                                              |
| `IOError` e `IOException`    | Grave errore di input o output                                                                                     |
| `IllegalArgumentException`   | Parametro illegale come input di un metodo                                                                         |
| `NumberFormatException`      | Errore nel formato di un numero (estende la precedente)                                                            |

---
## Blocco try-catch
Il blocco try-catch consente di **catturare le eccezioni**

Nel blocco `try` si inseriscono tutte le **istruzioni dalle quali vengono sollevate le eccezioni** che vogliamo catturare
All’interno del blocco `catch` è necessario indicare il **tipo di eccezione da catturare** e specificare nel blocco le azioni da attuare a seguito dell’eccezione sollevata (è possibile specificare molteplici blocchi catch, in risposta a differenti eccezioni sollevate)

```java
public class Spogliatoio {
	public static void main(String[] args) {
		Sportivo pellegrini = new Sportivo("Federica Pellegrini");
		Sportivo bolt = new Sportivo("Usain Bolt")
		Armadietto armadietto = new Armadietto(pellegrini);
		
		try {
			armadietto.apriArmadietto(bolt);
		}
		catch(NonToccareLaMiaRobaException e) {
			// provvedimenti contro i ladri
		}
		catch(ArmadiettoGiaApertoException e) {
			// notifica che l'armadietto è già aperto
		}
	}
}
```

E’ molto importante considerare l’ordine con cui si scrivono i diversi blocchi catch e catturare le eccezioni **dalla più specifica a quella più generale**. Nell’attuare il processo di cattura, la JVM sceglie il **primo catch compatibile**, tale cioè che il tipo dell’eccezione dichiarata sia lo stesso o un supertipo dell’eccezione lanciata durante l’esecuzione.
Spesso vogliamo rispondere ad un’eccezione con il rimedio specifico e non con uno più generale.

Ricorda bene che **l’ordine conta**
```java
public class TavoloDiGioco {
	public enum Seme {
		SPADE, DENARI, BASTONI, COPPE;
	}
	
	public static void main(String[] args) {
		try {
			// solleva eccezione più specifica 
			// (ovvero NumberFormatExceptionche che estende
			// IllegalArgumentException)
			Integer due = Integer.parseInt("Due");
			// solleva eccezione più generale
			Seme denari = Seme.valueOf("DENARA");
			
			Carta dueDiDenari = new Carta(due, denari);
		}
		catch(IllegalArgumentException e) {
			System.out.println("Seme non esistente");
		}
		// questo secondo caso non viene mai raggiunto
		catch(NumberFormatExceptionche e1) {
			System.out.println("Valore non esistente");
		}
	}
}
```

### Catch integrato di eccezioni alternative
E' possibile specificare un'unica clausola catch con diversi tipi di eccezioni utilizzando l'operatore `|`. Risulta tile laddove, a fronte di eccezioni diverse, si debba avere un comportamento analogo

```java
try {
	if (condizione) throw new Eccezione1();
	else throw new Eccezione2();
}
catch(Eccezione1|Eccezione2 e) {
	// gestione dei due casi in un unico blocco
}
```

### Flusso in presenza o assenza di eccezioni
Se durante l’esecuzione non vengono sollevate eccezioni:
1. ciascuna istruzione all’interno del blocco try viene eseguita normalmente
2. terminato il blocco try, l’esecuzione riprende dalla prima linea dopo il blocco try-catch

Se viene sollevata un’eccezione:
1. L’esecuzione del blocco try viene interrotta
2. Il controllo passa al primo blocco catch compatibile, tale cioè che il tipo dichiarato nella clausola catch sia dello stesso tipo dell’eccezione sollevata, o un suo super-tipo
3. L’esecuzione riprende dalla prima linea dopo il blocco try-catch

---
## La politica catch-or-declare
Una volta sollevata un’eccezione, possiamo:
- **Ignorare** l’eccezione e propagarla al metodo chiamante, a patto di aggiungere all’intestazione del metodo la clausola `throws`, seguìto dall’elenco delle eccezioni potenzialmente sollevate (*declare*)
- **Catturare** l’eccezione, ovvero gestire la situazione anomala in modo opportuno, prendendo provvedimenti e contromisure atte ad arginare il più possibile la situazione di emergenza (*catch*)

Se il requisito catch-or-declare non viene soddisfatto il compilatore emette un errore che indica che l’eccezione dev’essere **catturata** o **dichiarata**. Questo serve a **forzare il programmatore** a considerare i problemi legati all’uso di metodi che emettono eccezioni

### Ignorare le eccezioni
Se intendiamo ignorare l’eccezione siamo costretti a dichiarare esplicitamente il suo sollevamento con throws

```java
public class Spogliatoio {
	public static void main(String[] args) throws NonToccareLaMiaRobaException, ArmadiettoGiaApertoException {
		Sportivo pellegrini = new Sportivo("Federica Pellegrini");
		Sportivo bolt = new Sportivo("Usain Bolt")
		
		Armadietto armadietto = new Armadietto(pellegrini);
		armadietto.apriArmadietto(bolt)
	}
}
```

Il costrutto `throws` dichiara che il metodo (o i metodi delle classi da questo invocati) può sollevare eccezioni dello stesso tipo (o di un tipo più specifico) di quelle elencate dopo il `throws` (tale specifica non è sempre obbligatoria, ma dipende dal tipo di eccezione sollevata)

Se **tutti i metodi** all’interno dell’albero delle chiamate dell’esecuzione corrente decidono di ignorare l’eccezione, l’esecuzione viene **interrotta**. Questo però risulta vero solo del caso di applicazione a singolo thread (nel caso di molteplici thread, è il singolo thread ad essere interrotto; l’applicazione termina se sono interrotti tutti i thread)

![[Screenshot 2024-04-23 alle 19.05.00.png|370]]
Il metodo più in basso nello stack di attivazione lancia un’eccezione (*throw point*)
Tutti i metodi decidono di ignorare l’eccezione con throws
Come effetto si ha la **terminazione** dell’esecuzione

---
## I metodi printStackTrace() e getMessage()
Quando un’eccezione non viene mai catturata, l’effetto è il seguente:
```java
Exception in thread "main" NonToccareLaMiaRobaException 
at Armadietto.apriArmadietto(Armadietto.java:11)
at Spogliatoio.main(Spoiatoio.java:10)
```

Su schermo viene stampato un ‘riassunto’ associato all’eccezione non catturata, chiamato *stack trace*
Questo riporta:
- il **thread** in cui l’eccezione è stata sollevata
- il **nome** dell’eccezione sollevata
- la successione, in ordine inverso di invocazione, dei **metodi coinvolti**
- il **file sorgente** e il **numero di riga** di ciascuna invocazione

L’output generato a schermo da un’eccezione non catturata è prodotto dal metodo `printStackTrace()`, offerto dalla classe **Throwable**
Un altro metodo messo a disposizione dalla stessa classe è `getMessage()`, in grado di restituire, se prevista o disponibile, una della **descrizione sintetica** ragione per la quale si è verificata l’eccezione

---
## Eccezioni personalizzate
E’ possibile definire delle **eccezioni personalizzate** in modo tale da mantenere negli errori una semantica legata all’applicazione.
Al momento della creazione di un nuovo tipo di eccezione sarà opportuno **studiarne la natura** e lo scopo (errore sincrono/asincrono, possibilità di recovery o errore irreversibile...) e scegliere la super-classe più adeguata

Vediamo la definizione dell’eccezione personalizzata `NonToccareLaMiaRobaException`:
```java
public class NonToccareLaMiaRobaException extends Exception {

}
```
Tramite la parola chiave `extends` è possibile **creare** una nuova eccezione a partire da un tipo già esistente

```java
public class Armadietto {
	private Sportivo proprietario;
	private boolean aperto;
	
	public Armadietto(Sportivo proprietario) {
		this.proprietario = proprietario;
	}
	public void apriArmadietto(Sportivo g) throws NonToccareLaMiaRobaException, ArmadiettoGiaApertoException {
		if (!proprietario.equals(g)) throw new NonToccareLaMiaRobaException();
		if (aperto) throw new ArmadiettoGiaApertoException();
	}
}
```
Tramite la parola chiave `throw` è possibile **sollevare** (o lanciare) una nuova eccezione

---
## Blocco finally
E’ uno speciale blocco posto dopo tutti i blocchi try-catch ed è eseguito a prescindere dal sollevamento di eccezioni
Le istruzioni all’interno del blocco finally vengono **sempre eseguite** (perfino se nel blocco try-catch vi è un `return`, un `break` o un `continue`)

Tipicamente all’interno del blocco finally vengono eseguite operazioni di *clean-up* (es. chiusura di eventuali file aperti o rilascio di risorse) in modo da garantire un certo stato dell’esecuzione

```java
public class FileAperto {
	FileReader fileReader = null;
	
	try {
		fileReader = new FileReader(new File("my/favourite/path"));
		fileReader.read();
	}
	catch (FileNotFoundException e) { e.printStackTrace(); }
	catch (IOException e1) { e1.printStackTrace(); }
	finally {
		try {
			// clean-up
			fileReader.close();
		}
		catch (IOException e) { e.printStackTrace(); }
	}
}
```

---
## Classe Throwable
La classe che implementa il concetto di eccezioni è **Throwable** che estende direttamente la classe Object. Gli oggetti di tipo Throwable sono gli unici oggetti che è possibile utilizzare con il meccanismo delle eccezioni

![[Screenshot 2024-04-23 alle 19.38.04.png|600]]

### Classi Exception ed Error
**Exception**
- eccezioni interne alla JVM (classe `RuntimeException`) → legate ad errori nella logica del programma
- eccezioni regolari (es. `IOException`, `ParseException`, `TimeoutException`) → errori che le applicazioni dovrebbero anticipare e dalle quali poter riprendersi
**Error**: cattura l’idea di condizione eccezionale irrecuperabile
- Assai rari e non dovrebbero essere considerati dalle applicazioni (es. ThreadDeath, OutOfMemoryError...)

---
## Eccezioni checked e unchecked
### Checked
- È sempre necessario attenersi al paradigma catch-or-declare
- Sono eccezioni comuni, ovvero quelle che estendono `Exception` (ma non `RuntimeException`)
- Esempi: `ParseException`, `ClassNotFoundException`, `FileNotFoundException`

### Unchecked
- Non si è obbligati a dichiarare le eccezioni sollevate o a catturarle in un blocco try-catch (ma è possibile farlo)
- Sono eccezioni che estendono `Error` o `RuntimeException`
- Esempi: `IndexOutOfBoundsException`, `ClassCastException`, `NullPointerException`, `ArithmeticException`, `OutOfMemoryError`