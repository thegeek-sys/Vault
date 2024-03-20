---
Created: 2024-03-17
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Dichiarazione|Dichiarazione]]
	- [[#Introduction#Creazione senza valori|Creazione senza valori]]
	- [[#Introduction#Creazione con valori|Creazione con valori]]
- [[#Esempi di dichiarazione|Esempi di dichiarazione]]
- [[#Accedere agli elementi di un array|Accedere agli elementi di un array]]
- [[#Modificare array|Modificare array]]
- [[#Metodi con numero di parametri variabile|Metodi con numero di parametri variabile]]
- [[#Array a due dimensioni|Array a due dimensioni]]
---
## Introduction

Un array rappresenta un gruppo di variabili (chiamate elementi) tutte dello stesso tipo. Gli array sono oggetti, quindi **le variabili di array contengono il riferimento all’array**
Gli elementi di un array possono essere di due tipi:
- primitivi (interi, double, ecc.)
- riferimenti a oggetti (inclusi altri array)

### Dichiarazione
Dichiarandola in questo modo `a` in questo momento sarà `null` (non ci sta inizializzazione implicita di oggetti)
```java
int[] a;
```

### Creazione senza valori
In questo modo ogni elemento viene inizializzato con il valore di default del rispettivo tipo (0, false, null)
```java
a = new int[10]
```

### Creazione con valori
```java
a = new int[] { 5, -2, 3, 0, 1, -6, 75, 32, 122, 4 }
```

> [!warning]
> Non posso specificare dimensione e allo stesso tempo inizializzare i valori

---
## Esempi di dichiarazione

```java
int[] numeri = new int[10];


final int NUMERO_DI_CIFRE = 10;
int[] numeri = new int[NUMERO_DI_CIFRE];


int numeroDiCifre = new Scanner(System.in).nextInt();
int[] numeri = new int[numeroDiCifre];


// array di 10 interi costruito specificando i valori
int[] numeri = { 0,1,2,3,4,5,6,7,8,9 };


// array di 3 riferimenti a stringa,
// tutti inizializzati a null
String[] nomi = new String[3]


// array di 3 riferimenti a stringa, con valori assegnati
String[] nomi = { "mario", "luigi", "wario" }
```

---
## Accedere agli elementi di un array
Si accede a un elemento dell’array specificando il nome dell’array seguito dalla posizione (indice) dell’elemento tra parentesi quadre

```java
// stampa il sesto elemento dell'array
System.out.println(a[5]);

// memorizza la somma dei primi 3 elementi
int k = a[0] + a[1] + a[2];
```

L’indice è **sempre positivo** compreso tra 0 e dimensione dell’array-1 (non posso accedere all’ultimo elemento dell’array utilizzando indice negativo come in Python). L’indice può anche essere un’espressione `a[i+j*2]+=2`

E’ inoltre importante ricordare che essendo la lunghezza del vettore un valore memorizzato nel vettore stesso per accedervi ci basta usare `vettore.length` (non viene calcolato ogni volta come invece è per le stringhe)

---
## Modificare array
Un array ha dimensioni prefissate che NON possono essere modificate. Tuttavia è possibile creare un nuovo array con nuove dimensioni a partire da un array preesistente attraverso il metodo statico **`copyOf`** della classe `java.util.Arrays`

```java
import java.util.array

public class MyArrays {
	public static void main(String[] args) {
		// array di dimensione 9
		int[] array = { 1, 5, 8, 2, 3, 4, 7, 6, 9 };
		
		// restringe l'array a dimensione 5
		array = Arrays.copyOf(array, 5);
		System.out.println(Arrays.toString(array)); // [1, 5, 8, 2, 3]
		
		// allarga l'array a dimensione 8 (gli ultimi 3 valori
		// sono inizializzati a zero)
		array = Array.copyOf(array, 8);
		System.out.println(Arrays.toString(array)); // [1,5,8,2,3,0,0,0]
	}
}
```

---
## Metodi con numero di parametri variabile
Si possono dichiarare metodi con un numero variabile di parametri mediante la sintassi: `tipo...`

```java
public class SommaDouble {
	// metodi statici in modo tale da poteli chiamare da un'altra classe
	// senza dover istanziare un nuovo oggetto SommaDouble
	public static double sum(double... valori) {
		double somma = 0.0;
		for (int k=0; k<valori.length; k++) somma += valori[k];
		return somma;
	}
	
	public static double sumFirstN(final int N, double... valori) {
		double somma = 0.0;
		for (int k=0; k<valori.length && k<N; k++) somma += valori[k];
		return somma
	}
	
	public static void main(String[] args) {
		System.out.println(sum());
		System.out.println(sum(1,2,3,4));
		System.out.println(3, 1,2,3,4);
	}
}
```

---
## Array a due dimensioni
Si può specificare un array a 2 dimensioni (o matrice) semplicemente specificando due coppie di parentesi quadre
```java
String[][] matrice = new String[RIGHE][COLONNE];
```

L’accesso avviene specificando le due dimensioni dell’array
```java
System.out.println(matrice[x][y]);
```
