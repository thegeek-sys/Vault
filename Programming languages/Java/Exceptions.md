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