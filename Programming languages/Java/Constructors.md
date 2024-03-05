---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

- [[#Introduzione|Introduzione]]
- [[#Esempio|Esempio]]
	- [[#Esempio#Costruttore di classe|Costruttore di classe]]
	- [[#Esempio#Creazione oggetto|Creazione oggetto]]
---
## Introduzione
I costruttori sono dei metodi speciali che mi permettono di creare degli oggetti di una classe e hanno lo stesso nome della classe.
Questi inizializzano i campi dell’oggetto e possono prendere zero, uno o più parametri.

> [!warning]
> **Non** hanno valori di uscita ma **non** specificano void

Una classe può avere più costruttori (con lo stesso nome dell’originale) che differiscono per numero e tipi dei parametri (sto facendo [[Classes#Esercizio contatore|overloading]] del costruttore).
Se non viene specificato un costruttore, Java crea per ogni classe un costruttore di default “vuoto” (senza parametri) che inizializza le variabili d’istanza ai valori di default (int inizializzati a zero).

---
## Esempio
### Costruttore di classe
```java
public class Counter {
	// Costruttore
	public Counter() {
		valore = 0;
	}
}
```

### Creazione oggetto
```java
static public void main(String[] args) {
	Counter contatore1 = new Counter()
}
```