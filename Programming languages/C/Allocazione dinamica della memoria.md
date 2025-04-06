---
Created: 2025-04-07
Programming language: "[[C]]"
Class: "[[Sistemi Operativi]]"
Related:
---
---
## Puntatori
Diverse delle funzioni fino ad ora usate utilizzano i puntatori (es. `strcpy`, `scanf`)
Un **puntatore** è una variabile che contiene l’indirizzo di un’altra variabile, infatti invece di contenere direttamente un valore punta a dove quel valore è memorizzato in memoria

```c
<type> *var_name;
```

>[!example]
>Ad esempio `char *prt` ha due tipi di valori:
>- **valore diretto**
>- **valore indiretto**
>
>Come valore diretto ha l’indirizzo di una cella di memoria a cui si accede tramite il nome della variabile
>Come valore indiretto ha il valore contenuto dalle celle di memoria a cui punta la variabile

### Operatori $\verb|&|$ e $\verb|*|$
L’operatore `*` è noto come operatore di dereferenziazione e serve per accedere al valore contenuto nell’indirizzo a cui un puntatore sta puntando
Invece l’operatore `&` serve per ottenere l’indirizzo di memoria di una variabile

Spesso questi due operatori vengono usati insieme:
```c
int num=5;
int *numPtr;

numPtr=&num; // assegna a numPtr l'indirizzo di num
*numPtr=10; // assegna alla locazione puntata da numPtr il valore 10
*numPtr=*numPtr+1 // assegna alla locazione di numPtr il valore numPtr+1
```

Usiamo questi operatori inoltre quando abbiamo la necessità di modificare il valore di una variabile all’interno di una funzione
```c
int main() {
	int n;
	int nprt;
	n=5;
	nprt=&n;
	increment(nprt); // oppure increment(&n);
	incrementVal(n); // oppure incrementVal(*nptr);
}

void increment(int *num) {
	// modifica il valore di num
	*num += 1; // o (*num)++;
}
void incrementVal(int num) {
	// NON modifica il valore di num
	num += 1; // o (*num)++;
}
```

### Vettori e puntatori
Il p