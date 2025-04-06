---
Created: 
Class: "[[Sistemi Operativi]]"
Programming language: "[[C]]"
Related:
---
---
## Introduzione
Le **structures** sono un potente strumento che permette di raggruppare insieme variabili di diversi tipi sotto un unico nome. Una struttura può contenere dati di tipi diversi e risulta particolarmente utile per organizzare i dati in modo chiaro e strutturato
Queste inoltre devono essere definire prima di essere usate

Si hanno tre modi per definirle
- **variabile struttura** (*variable*)
- **struttura taggata** (*taggged*)
- **definizione di tipo di struttura** (*type-defined*)

---
## Variabile struttura

```c
struct {
	double x; // coordinata x
	double y; // coordinata y
} point2d; // nome della variabile
```

Questa definizione però risulta poco portabile, infatti tutte le variabili di struttura vanno definite insieme. Se invece vogliamo creare un vero e proprio tipo di dato nuovo (la dobbiamo usare più volte) è meglio usare la tagged structure

---
## Tagged structure

```c
struct point3d { // tag della struttura
	double x; // coordinata x
	double y; // coordinata y
	double z; // coordinata z
};

struct point3d pointA, pointB, pointC;
```

Questa soluzione è più portabile, potrei infatti mettere la definizione di `struct point3d` in un header file e poi riutilizzarla

---
## Type-defined structure

```c
typedef struct {
	char ID[17]; // codice fiscale
	long int income;
	float taxRate;
} taxpayer_t; // nome del nuovo tipo di dato

taxpayer_t person1, persone2;
taxpayer_t persons[100];
```

Il **`typedef`** consente di creare un nuovo nome per un tipo di dato già esistente. È utile per ridurre la complessità delle dichiarazioni di variabili e migliorare la leggibilità del codice, senza alterare il comportamento del programma.
In particolare viene spesso usato in combinazione con `struct` per “far finta” di creare un nuovo dato

