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


