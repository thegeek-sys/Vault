---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduzione
I costruttori sono dei metodi speciali che mi permettono di creare degli oggetti di una classe e hanno lo stesso nome della classe.
Questi inizializzano i campi dell’oggetto e possono prendere zero, uno o più parametri.

> [!warning]
> **Non** hanno valori di uscita ma **non** specificano void

Una classe può avere più costruttori (con lo stesso nome dell’originale) che differiscono per numero e tipi dei parametri (sto facendo [[Classes#Esercizio contatore|overloading]] del costruttore).
Se non viene specificato un costruttore, Java crea per ogni classe un costruttore di default “vuoto” (senza parametri) che inizializza le variabili d’istanza ai valori di default (int inizializzati a zero).