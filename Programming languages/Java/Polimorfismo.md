---
Created: 2024-03-26
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Il polimorfismo è uno dei punti cardine della programmazione orientata agli oggetti oltre all’ereditarietà. Questo dunque ci permette di utilizzare un metodo senza dover conosce il tipo esatto (la classe) su cui si invoca il metodo

Una variabile di un certo tipo a può contenere un riferimento a un oggetto del tipo A o di qualsiasi sua sottoclasse
```java
// a è il riferimento di animale
// ma ho usato il costruttore di Gatto
Animale a = new Gatto();
a = new Chihuahua();
```

La selezione del metodo da chiamare avviene in base all’effettivo tipo dell’oggetto riferito alla variabile
```java
Animale a = new Gatto();
a.emettiVerso(); // "miaoo"
a = new Chihuahua();
a.emettiVerso(); // "bau bau"
```

---
## Binding
Per **binding** in programmazione si intende associare ad ogni variabile il proprio tipo
### Statico
Il **binding statico** consiste nell’associare una variabile al proprio tipo, e viene svolto in java dal compilatore che creerà quindi una tabella dei binding statici
Questo viene fatto, in Java così negli altri linguaggi compilati senza eseguire il codice ma solo “osservandolo”

### Dinamico
Il polimorfisrmo, come implementato in java, vede la JVM elaborare il **binding dinamico**, poiché l’associazione tra una variabile di riferimento e un metodo da chiamare viene stabilita a tempo di esecuzione.
Questo viene solitamente utilizzato dai linguaggi interpretati (come Python) e in Java viene utilizzato quando, attraverso il polimorfismo, utilizzo il costruttore di una sottoclasse del tipo di definizione oppure quando chiamo dei metodi

---
## Esempio

> [!hint]
> `@Override` serve a noi programmatori per chiedere al compilatore se esiste una classe superiore con uno stesso metodo. Se ciò non accade mi viene restituito un errore, e vuol dire quindi che non sto facendo alcun tipo di overriding
> 
> Posso anche chiamare un metodo di un supercostruttore attraverso `super.metodo()` all’interno di un overriding in una sottoclasse. In questo caso il binding viene fatto in modo dinamico (non lo riesco a capire leggendo direttamente il codice ma devo chiamare il costruttore superiore)

```java
'StringaHackerata.java'
import java.util.Random;

public class StringaHackerata {
	private String s;
	
	public StringaHackerata(String s) {
		this.s = s;
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		Random random = new Random();
		
		for (int k=0; k<s.lenght(); k++) {
			char c = s.charAt(k);
			if (random.nextBoolean()) c = Character.toUpperCase(c);
			else c = Character.toLowerCase(c);
			
			sb.append(c)
		}
		
		return sb.toString();
	}
}


'StringaHackerataConStriscia.java'
import java.util.Random;

public class StringaHackerataConStriscia extends StringaHackerata {
	final public static int MAX_LUNGHEZZA = 10;
	
	public StringaHackerataConStriscia(String s) {
		super(s);
	}
	
	public String getStriscia() {
		Random random = new Random();
		int len = random.nextInt(MAX_LUNGHEZZA);
		StringBuffer sb = new StringBuffer();
		
		// -=-=-=-
		for (int k=0; k<len; k++) sb.append(k%2 == 0 ? '-' : '=');
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		String striscia = getStriscia();
		// chiamo il metodo toString del supercostruttore
		return strisciaè+" "+super.toString()+" "+striscia;
	}
}

'TestStringa.java'
public static void main(String[] args) {
	StringaHackerata s1 = new StringaHackerata("drago di java")
	StringaHackerataConStriscia s2 = new StringaHackerataConStriscia("")
}
```

> [!info]
> Quando chiamo il print su un oggetto in automatico viene chiamato il relativo `.toString()`

