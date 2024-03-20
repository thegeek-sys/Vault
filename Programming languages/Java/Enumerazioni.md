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