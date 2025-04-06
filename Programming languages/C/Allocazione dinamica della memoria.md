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
Il puntatore al primo elemento il puntatore al vettore sono la stessa cosa, inoltre, dato che i vettori sono allocati in celle di memoria continue, si può iterare sul puntatore usando il suo stesso indirizzo
```c
int vect[10];
int *ptr=NULL;
ptr = &vect[0]; // putatore al primo elemento
ptr = vect;     // puntatore al vettore

i=0;
// assegna a vect i valori da 0 a 10
do{
	*ptr = i;
	ptr++; i++;
} while (ptr<=&vect[10]);
```

---
## Allocazione dinamica
Mentre vettori e altre variabili sono allocate nello stack a tempo di compilazione, esistono dei modi per allocare memoria a runtime (viene allocata nell’heap).
Per farlo usiamo i comandi:
```c
void *calloc(size_t nmemb, size_t size);
void *malloc(size_t size);
void free(void *prt)
```

### $\verb|calloc|$
`calloc` riserva spazio di memoria per un array di `size` elementi, ciascuno di dimensione `nmemb` byte; inoltre questa funzione inizializza tutti bit a $0$

> [!example]
> ```c
> char *a;
> const int SIZE_OF_ARRAY=30;
> a = (char *) calloc(SIZE_OF_ARRAY, sizeof(char));
> ```
>
>>[!question] Perché si usa il casting?
>>Il casting è principalmente una convenzione derivante dal fatto che queste funzioni restituiscono un **puntatore generico di tipo `void*`**, mentre in C è **preferibile specificare il tipo di puntatore** (es. `int*`, `char*`, ecc.) per evitare confusione, migliorare la leggibilità del codice e per compatibilità con C++

### $\verb|malloc|$
`malloc` riserva spazio di memoria di `size` byte. A differenza di `calloc` non inizializza gli elementi a $0$

### $\verb|free|$
`free` serve a liberare il blocco di memoria `prt`