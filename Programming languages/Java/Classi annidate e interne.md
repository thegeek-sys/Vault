---
Created: 2024-04-10
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Le classi usate finora vengono dette **top-level**, cioè esse si trovano più in alto di tutte le altre e non sono contenute in altre classi
Questo tipo di classi richiede un file `.java` dedicato con lo stesso nome della classe che esso contiene

---
## Classi annidate (nested class)
Java consente di scrivere classi all’interno di altre classi. Le classi presenti all’interno sono chiamate classi annidate (nested classe). Queste possono essere di due tipi:
- **static**
- **non-static** → in questo caso vengono dette **classi interne** (inner class)

### Classi interne (inner class)
Prima di poter creare un oggetto della classe interna è necessario istanziare la classe esterna (top-level) che la contiene. Ciascuna classe interna, infatti, ha un riferimento implicito all’oggetto della classe che la contiene.

Dalla classe interna è **possibile accedere a tutte le variabili e a tutti i metodi della classe esterna**. Inoltre come tutti i membri di una classe, le classi interne possono essere dichiarate public, protected o private

Per disambiguare casi di ambiguità (es. campi con lo stesso nome nella classe interna ed esterna):
- Se dalla classe interna viene usato soltanto `this` si fa riferimento ai campi e ai metodi di quella classe
- Per far riferimento alla classe esterna è necessario **precedere `this` dal nome della classe esterna e il punto**

Per istanziare la classe interna dalla classe esterna è sufficiente utilizzare l'operatore `new`. Per istanziare la classe interna da un’altra classe si utilizza la sintassi `riferimentoOggettoClasseEsterna.new ClasseInterna()`

#### Esempio
```java
public class Tastiera {
	private String tipo;
	private Tasto[] tasti;
	
	public class Testo {
		private char c;
		
		public Tasto(char c) {
			this.c = c;
		}
		public char premi() {
			return c;
		}
		public String toString() {
			// Tasto ha accesso al campo (privato!) della 
			// classe esterna
			// Posso anche accedere al campo direttamente tramite
			// il nome (tipo, invece di Tastiera.this.tipo)
			return Tastiera.this.tipo + ": " + premi();
		}
	}
	
	public Tastiera(String tipo, char[] caratteri) {
		this.tipo = tipo;
		tasti = new Tasto[caratteri.length];
		
		for (int i=0; i<caratteri.length; i++) {
			tasti[i] = new Tasto(caratteri[i]);
		}
	}
}
```
