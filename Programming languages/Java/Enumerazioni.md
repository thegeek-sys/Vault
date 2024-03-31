---
Created: 2024-03-20
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Struttura|Struttura]]
	- [[#Struttura#Esempio|Esempio]]
- [[#values() e valueOf()|values() e valueOf()]]
	- [[#values() e valueOf()#Esempio|Esempio]]
- [[#Enumerazioni e switch|Enumerazioni e switch]]
- [[#Enumerazioni e Object]]
	- [[#Enumerazioni e Object#Esempio|Esempio]]
---
## Introduction
Spesso è utile definire dei tipi (detti enumerazioni) i cui valori possono essere scelti tra un insieme predefinito di identificatori univoci (ogni identificatore corrisponde a una costante implicitamente static).
Non è possibile creare un oggetto del tipo enumerato

> [!warning]
> Ricorda di usare l’enumerazione ogni qualvolta ciò è possibile (preferire enumerazioni a array)

---
## Struttura
Un tipo enumerativo viene dichiarato mediante la sintassi:
```java
public enum NomeEnumerazione {
	COSTANTE1, COSTANTE2, ..., COSTANTEN
}
```

Come tutte le classi, la dichiarazione di una enumerazione può contenere altre componenti tradizionali:
- costruttori
- campi
- metodi

E’ importante ricordare che **il costruttore viene chiamato dal compilatore** quando viene eseguito (non lo posso richiamare io)

### Esempio
```java
'''
SENZA ENUMERAZIONI
'''
public class Mese {
	private int mese;
	
	public Mese(int mese) { this.mese = mese; }
	
	public int toInt() { return mese; }
	public String toString() {
		switch(mese) {
			case 1 -> return "GEN";
			case 2 -> return "FEB";
			/* ... */
			case 12 -> return "DIC";
			default -> return null;
		}
	}
}

'''
CON ENUMERAZIONI
'''
public enum Mese {
	GEN(1), FEB(2), MAR(3), APR(4), MAG(5), GIU(6),
	LUG(7), AGO(8), SET(9), OTT(10), NOV(11), DIC(12);
	private int mese;
	
	// costruttore con visibilità di default
	Mese(int mese) { this.mese = mese; }
	public int toInt() { return mese; }
}

Mese.GEN.toInt() // 2
```

---
## values() e valueOf()
Per ogni enumerazione, il compilatore genera i metodi statici:
- `values()` → che restituisce un array delle costanti enumerative
- `valueOf()` → che restituisce la costante enumerativa associata alla stringa fornita in input (se il valore non esiste viene emessa un’eccezione)

### Esempio
```java
public enum SemeCarta {  
    CUORI, QUADRI, FIORI, PICCHE;  
  
    public static void main(String[] args) {  
        SemeCarta[] valori = SemeCarta.values();  
        for (int k = 0; k < valori.length; k++) {  
            System.out.print(valori[k]+" "); // CUORI QUADRI FIORI PICCHE
        } System.out.println();
        
        String v = "PICCHE";  
        SemeCarta picche = SemeCarta.valueOf(v);  
        System.out.println(picche); // PICCHE
    }  
}
```

---
## Enumerazioni e switch
Le enumerazioni possono essere utilizzate all’interno di un costrutto `switch`

```java
SemeCarta seme = null;

/* ... */

switch(seme) {
	case CUORI: System.out.println("Come"); break;
	case QUADRI: System.out.println("Quando"); break;
	case FIORI: System.out.println("Fuori"); break;
	case PICCHE: System.out.println("Piove"); break;
}
```

---
## Enumerazioni e Object
Una enumerazione ha tante istanze quante sono le costanti enumerative a suo interno. Non è possibile infatti costruire altre istanze, ma possono essere costruite le istanze “costanti”:
- si definisce un costruttore (non pubblico, ma con visibilità di default)
- si costruisce ciascuna costante (un oggetto separato per ognuna)
- si possono definire altri metodi di accesso o modifica dei campi

Le classi enumerative estendono la classe `Enum`, da cui ereditano i metodi `toString` e `clone`:
- `toString()` → restituisce il nome della costante
- `clone()` → restituisce l’oggetto enumerativo stesso senza farne una copia (che non è possibile fare visto che sono costanti)

`Enum` a sua volta estende `Object`, per cui il metodo `equals` restituisce true solo se le costanti enumerative sono identiche

```java
public enum TipoDiMoneta {
	// le costanti enumerative, costruite in modo appropriato
	CENT(0.01), CINQUE_CENT(0.05), DIECI_CENT(0.10), VENTI_CENT(0.20),
	CINQUANTA_CENT(0.50), EURO(1.00), DUE_EURO(2.00);
	
	// valore numerico della costante
	private double valore;
	
	// costruttore con visibilità di default
	TipoDiMoneta(double valore) { this.valore = valore; }
	
	// meotodo di accesso al valore
	public double getValore() { return valore; }
}
```

### Esempio
```java
public enum Pianeta {
	MERCURIO (3.303e+23, 2.4397e6),   VENERE   (4.869e+24, 6.0518e6),
	TERRA    (5.976e+24, 6.37814e6),  MARTE    (6.421e+23, 3.3972e6),
	GIOVE    (1.9e+27,   7.1492e7),   SATURNO  (5.688e+26, 6.0268e7),
	URANO    (8.686e+25, 2.5559e7),   NETTUNO  (1.024e+26, 2.4746e7);
	
	// costante di gravitazione universale
	public static final double G = 6.67300E-11;
	// massa in kg
	private final double massa;
	// raggio in metri
	private final double raggio;
	
	Pianeta(double massa, double raggio) {
		this.massa = massa;
		this.raggio = raggio;
	}
	
	private double getMassa() { return massa; }
	private double getRaggio() { return raggio; }
	public double getGravitaDiSuperficie() {
		return G * massa / (raggio * raggio);
	}
	public double getPesoDiSuperficie(double altraMassa) {
		return altraMassa * getGravitaDiSuperficie();	
	}
}
```