---
Created: 2024-03-20
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
L’**ereditarietà** è un concetto cardine della programmazione orientata agli oggetti. E’ una forma di riuso del software in cui una classe è creata:
- “assorbendo” i membri di una classe esistente
- aggiungendo nuove caratteristiche o migliorando quelle esistenti
Questo tipo di organizzazione aumenta le probabilità che il sistema sia **implementato e mantenuto in maniera efficiente**

Per esempio potremmo progettare una classe `Forma` che rappresenta una forma generica e poi **specializzarla estendendo** la classe
![[Screenshot 2024-03-20 alle 19.07.46.png|center|600]]

```java
'Forma.java'
public class Forma {
	public void disegna() {  }
}


'Triangolo.java'
public class Triangolo extends Forma {
	private double base;
	private double altezza;
	
	public Triangolo(double base, double altezza) {
		this.base = base;
		this.altezza = altezza;
	}
	
	public double getBase() { return base; }
	public double getAltezza() { return altezza; }
}


'Cerchio.java'
public class Cerchio extends Forma {
	private double raggio;
	
	public Cerchio(int raggio) { this.raggio = raggio; }
	
	public double getRaggio() { return raggio; }
	public double getCirconferenza() { return 2*Math.PI*raggio; }
}
```

---
## Cosa si eredita?
Una **sottoclasse** estende la **superclasse** (una sottoclasse può avere al più una superclasse) ereditando così i membri della superclasse (campi e metodi d’istanza secondo il livello di accesso specificato).
Inoltre la sottoclasse può:
- **aggiungere** nuovi metodi e campi
- **ridefinire** i metodi che eredita dalla superclasse (tipicamente NON i campi)

![[Screenshot 2024-03-20 alle 19.19.27.png|center|350]]

---
## Classi astratte
Una classe **astratta** (definita mediante la parola chiave `abstract`) non può essere istanziata; il che vuol dire che NON possono esistere oggetti per quella classe.
Anche i metodi possono essere definiti astratti ma esclusivamente all’interno di una classe dichiarata astratta. Impongono alle sottoclassi non astratte di implementare il metodo.

```java
public abstract class PersonaggioDisney {
	// metodo astratto senza implementazione
	abstract void faPasticci();
}

// non posso fare PersonaggioDisney a = new PersonaggioDisney();
// non è possibile istanziarla
```

Tipicamente verrà estesa da altre classi, che invece potranno essere istanziate
```java
public class Paperoga extends PersonaggioDisney {
	public void faPasticci() {
		System.out.println("bla bla bla bla bla");
	}
}
```

Ereditando da una classe astratta dei metodi astratti, a meno che non sono astratto anche io, devo necessariamente definire quel metodo.

La visibilità protetta (`protected`) rende visibile il campo (o il metodo) a tutte le sottoclassi (ma anche a tutte le classi del package)

![[Screenshot 2024-03-20 alle 19.29.48.png|center|650]]

---
## Esempio: Impossible Mission
Abbiamo tanti “oggetti”:
- Piattaforme
- Computer
- Oggetti in cui cercare indizi

Alcuni sono “personaggi”:
- Il giocatore
- I robot
- Il “bombone”

Abbiamo bisogno di una classe molto generale (quindi astratta) che rappresenti oggetti mobili e immobili nel gioco:
```java
public abstract class Entita {
	protected int x;
	protected int y;
	
	public Entita(int x, int y) {
		this.x = x;
		this.y = y;
	}
	
	public int getX() { return x; }
	public int getY() { return y; }
}
```

Modelliamo gli oggetti immobili in astratto:
```java
public abstract class Oggetto extends Entita {
	private TesseraPuzzle tessera;
	
	public Oggetto(int x, int y) {
		// se mi viene passato solo x e y chiamo l'altro costruttore
		// con null come terzo parametro
		this(x, y, null);
	}
	
	public Oggetto(int x, int y, TesseraPuzzle tessera) {
		// chiamo il costruttore della classe Entita
		// OBBLIGATORIO perché ha almeno un parametro
		super(x, y);
		this.tessera = tessera;
		
	}
	
	// metodo aggiuntivo
	public TesseraPuzzle search() { return tessera; }
}
```

Gli oggetti che possono contenere una tessera del puzzle del gioco:
```java
public class TesseraPuzzle {
	// da implementare
}
```

---
## this e super
La parola chiave `this` usata come nome di metodo **obbligatoriamente** nella prima riga del costruttore permette di richiamare un altro costruttore della stessa classe