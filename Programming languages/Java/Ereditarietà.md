---
Created: 2024-03-20
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Cosa si eredita?|Cosa si eredita?]]
- [[#Classi astratte|Classi astratte]]
- [[#this e super|this e super]]
	- [[#this e super#Esempio|Esempio]]
- [[#Overriding e Overloading|Overriding e Overloading]]
- [[#Visibilità|Visibilità]]
- [[#is-a vs. has-a|is-a vs. has-a]]
- [[#Esempio: Impossible Mission|Esempio: Impossible Mission]]

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
## this e super
La parola chiave `this` usata come nome di metodo **obbligatoriamente** nella prima riga del costruttore permette di richiamare un altro costruttore della stessa classe ([[#^this|esempio]])

La parola `super` usata come nome di metodo **obbligatoriamente** nella prima riga del costruttore permette di richiamare un costruttore della superclasse ([[#^this|esempio]])

> [!warning]
> Ogni sottoclasse deve esplicitamente definire un costruttore se la superclasse NON fornisce un costruttore senza argomenti (cioè la superclasse ha un costruttore con argomenti, vanno “mandati“ dalla sottoclasse con un costruttore)

### Esempio
```java
'X.java'
public class X {
	public X(int k) {
		System.out.println("X(int k)");
	}
	
	public X() {
		System.out.println("X()");
	}
}

'Y.java'
public class Y extends X {
	public Y(int k) {
		System.out.println("Y(int k)");
	}
}

'Z.java'
public class Z extends Y {
	public Z(int k) {
		super(k);
		System.out.println("Z(int k)");
	}
	
	public Z() {
		this(0);
		System.out.println("Z()");
	}
	
	public static void main(String[] args) {
		Z z = new Z();
	}
}


-> X()
-> Y(int k)
-> Z(int k)
-> Z()
'''
```

---
## Overriding e Overloading
L’**overriding** consiste nel ridefinire (reimplementare) un metodo con la stessa intestazione (“segnatura”) presente in una superclasse
- nell’overriding gli **argomenti devono essere gli stessi**
- i **tipi di ritorno devono essere compatibili** (lo stesso tipo o una sottoclasse). Posso dunque specializzare il tipo di ritorno del metodo affinché sia una sottoclasse della classe superiore (non posso ritornare un tipo di una classe superiore rispetto a quella da cui sto facendo l’overriding, ad es. se la superclasse ritorna un long posso ritornare un int ma non viceversa)
- **non si può ridurre la visibilità** (es. da public a private)

L’**overloading** consiste nel creare un metodo con lo stesso nome, ma una intestazione diversa (diverso numero e/o tipo di parametri)
- i **tipi di ritorno possono essere diversi**, ma non si può cambiare solo il tipo
- si può **variare la visibilità** in qualsiasi direzione

[[#^overloading-overriding|Qui]] si può trovare un esempio di utilizzo di overloading e overriding

---
## Visibilità
![[Screenshot 2024-03-25 alle 22.39.43.png]]
Abbiamo quattro possibilità per campi e metodi:
- `private` → visibile solo all’interno della classe
- `public` → visibile a tutti (all’interno di un modulo)
- `default` → visibile all’interno di tutte le classi del package
- `protected`→ visibile all’interno di tutte le classi del package e delle sottoclassi (indipendentemente dal package)

---
## is-a vs. has-a
E’ molto importante distinguere tra relazioni di tipo **is-a** (è-un) e relazioni di tipo **has-a** (ha-un)
**is-a** rappresenta l’*ereditarietà*. Un oggetto di una sottoclasse può essere trattato come un oggetto della superclasse. Domanda: la sottoclasse è-un superclasse? (es. Paperino è un PersonaggioDisney? Sì! QuiQuoQua è un Paperino? No!)
**has-a** rappresenta la *composizione*. Un oggetto contiene come membri riferimenti ad altri oggetti. Domanda: un oggetto contiene altri oggetti? (es. Bagno contiene Vasca? Sì! PersonaggioDisney contiene Paperino? No!)

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

Abbiamo bisogno di una classe molto generale (quindi astratta) che rappresenti **oggetti mobili e immobili** nel gioco:
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

Modelliamo gli **oggetti immobili** in astratto: ^this-super
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

Gli oggetti che possono contenere una **tessera** del puzzle del gioco:
```java
public class TesseraPuzzle {
	// da implementare
}
```

Modelliamo un possibile **oggetto immobile**:
```java
public class Libreria extends Oggetto {
	public Libreria(int x, int y, TesseraPuzzle tessera) {
		super(x, y, tessera);
	}
	
	public Libreria(int x, int y) {
		super(x, y);
	}
}
```

E il **computer**:
```java
public class Computer extends Entita {
	public Computer(int x, int y) {
		super(x, y);
	}
	
	public void login() { /* da implementare */ }
	public void logout() { /* da implementare */ }
}
```

Modelliamo un generico **personaggio**:
```java
public abstract class Personaggio extends Entita {
	public enum Direzione {
		DESTRA,
		SINISTRA,
		ALTO,
		BASSO;
	}
	
	private String nome;
	private int velocita;
	
	public Personaggio(int x, int y, String nome, int velocita) {
		super(x, y);
		this.nome = nome;
		this.velocita = velocita;
	}
	
	public String getNome() { return nome; }
	public int getVelocita() { return velocita; }
	
	public void muoviti(Direzione d) {
		switch(d) {
			case DESTRA: x+=velocita; break;
			case SINISTRA: x-=velocita; break;
			// in futuro: emetti eccezione!
			default: System.out.println("Direzione non ammessa"); break;
		}
	}
}
```

Modelliamo il **giocatore** (ovvero la spia):
```java
public class Spia extends Personaggio {
	public Spia(int x, int y, String nome, int velocita) {
		super(x, y, nome, velocita);
	}
	
	public void salta() {
		// ...
	}
}
```

Modelliamo un generico **nemico**:
```java
public abstract class Nemico extends Personaggio {
	public Nemico(int x, int y, String nome, int velocita) {
		super(x, y, nome, velocita);
	}
	
	public abstract void attacca();
}
```

E i **robot**:
```java
public class Robot extends Nemico {
	public Robot(int x, int y, String nome, int velocita) {
		super(x, y, nome, velocita);
	}
	
	public void incenerisci() { /* fulmine elettrico */ }
	// siamo obbligati a definire il metodo astratto
	public void attacca() { /* da implementare */ }
}
```

Modelliamo un **bombone**: ^overloading-overriding
```java
public class Bombone extends Nemico {
	public Bombone(int x, int y, String nome, int velocita) {
		super(x, y, nome, velocita);
	}
	
	public void attacca() {
		// da implementare
	}
	
	// OVERRIDING del metodo
	public void muoviti(Direzione d) {
		switch(d) {
			case DESTRA: case SINISTRA: super.muoviti(d); break;
			case ALTO: y-=getVelocita(); break;
			case BASSO: y+=getVelocita(); break;
		}
	}
	
	// OVERLOADING del metodo
	public void muoviti(Spia p) {
		muoviti(p.x > this.x ? Direzione.DESTRA : Direzione.SINISTRA);
		muoviti(p.y > this.y ? Direzione.ALTO : Direzione.BASSO)
	}
}
```
