---
Created: 2024-05-08
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Esempio|Esempio]]
- [[#Come funziona la ricorsione?|Come funziona la ricorsione?]]
- [[#Mutua ricorsione|Mutua ricorsione]]
	- [[#Mutua ricorsione#Esempio|Esempio]]

---
## Introduction
La ricorsione serve per risolvere problemi dei quali si può ridurre la dimensione fino ad arrivare a dei casi noti.
In particolar mod, dato un problema bisogna identificare:
- i casi base → problemi di dimensione minima e immediatamente calcolabile
- il passo ricorsivo → si parte dal problema principale e si cerca di ridurre la dimensione del problema

### Esempio
```java
public int fattorialeIterativo(int n) {
	int v = 1;
	for (int k=1; k<=n; k++) v *= k;
	return v
}


public int fattorialeRicorsivo(int v) {
	if (n == 0) return 1;
	return n*fattorialeRicorsivo(n-1);
}
```

---
## Come funziona la ricorsione?
Ogni volta che viene effettuata una chiamata a un metodo viene creato un **nuovo ambiente**, detto *record di attivazione*. Questo record di attivazione contiene la zona di memoria per le variabili locali del metodo e l’indirizzo di ritorno al metodo chiamante (a ogni chiamata, il record corrispondente viene **aggiunto sullo stack**).

>[!hint]
>Lo stack è la pila dei record di attivazione delle chiamate annidate

Cosa succede in memoria quando chiamiamo `fattorialeRicorsivo(5)`?
![[Screenshot 2024-05-07 alle 09.37.26.png|580]]

Però la ricorsione per come l’abbiamo implementata non gestisce valori `< 0`, il che significa che chiamando la ricorsione su valori negativi, il numero continuerà a decrementare fin quando non si riempirà lo stack e a quel punto mi verrà restituito uno `java.lang.StackOverflowError` (errore non gestibile).
Per ovviare a questo problema devo fare in modo di gestire il caso in cui mi venga dato in input un valore negativo e restituire un errore
```java
public class FattorialeFattoBene {
	public int fattorialeRicorsivo(int n) throws NumeroNonAmmessoException {
		if (n < 0) throw new NumeroNonAmmessoException;
		if (n == 0) return 1;
		return n*fattorialeRicorsivo(n-1);
	}
}


public class NumeroNonAmmessoException extends Exception {
	private int n;
	public NumeroNonAmmessoException(int n) {
		this.n = n;
	}
	public String toString() {
		return super.toString()+": "+n;
	}
}
```

---
## Mutua ricorsione
A volte può verificarsi il caso in cui il metodo `a` richiami `b` e `b`, a sua volta, richiami `a`. Questo meccanismo di reciproca chiamata dei metodi è detto di **mutua ricorsione**.
Ovviamente è sempre necessario stabilire **uno o più** casi base, onde evitare un ciclo infinito di chiamate tra `a` e `b`.

### Esempio
Scrivere una classe che, data una lista di interi, controlli ricorsivamente se ciascun numero in posizione pari sia dispari e ciascun numero in posizione dispari sia pari
```java
public class PariDispari {
	public boolean pariDispari(ArrayList<Integer> l) {
		return pariDispari(l, 0);
	}
	private boolean pariDispari(ArrayList<Integer> l, int k) {
		if (k == l.size()) return true;
		return (l.get(k) % 2 == 0) %% dispariPari(l, k+1);
	}
	private boolean dispariPari(ArrayList<Integer> l, int k) {
		if (k == l.size()) return true;
		return (l.get(k) % 2 == 1) && pariDispari(l, k+1);
	}
}
```
