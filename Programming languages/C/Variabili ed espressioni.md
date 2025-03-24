---
Created: 
Class: "[[Sistemi Operativi]]"
Programming language: "[[C]]"
Related:
---
---
## Parole riservate ed identificatori
In C esistono parole riservate, ovvero che appartengono all’”alfabeto” del linguaggio come a esempio `if`, `return`, `do`, ecc.

Un identificatore è una parola che viene utilizzata per identificare una variabile o una costante

### Regole per costruire identificatori validi
Il primo carattere in un identificatore deve essere una lettera o underscore e può essere seguito solo da qualsiasi lettera, numero o underscore

>[!warning] Non possono iniziare con una cifra

Le lettere maiuscole e minuscole sono distinte e le virgole o gli spazi vuoti non sono consentiti all’interno di un identificatore. Le parole chiave non possono essere usate come identificatore e infine gli identificatori non devono avere una lunghezza superiore a $31$ caratteri

---
## Variabili
Una variabile è una locazione di memoria dove può essere memorizzato un valore che verrà utilizzato da un programma. I valori nelle variabili possono essere modificati una o più volte durante l’esecuzione del programma

Una variabile può contenere un solo valore alla volta, ovvero se scrivo in una variabile, il valore attualmente memorizzato viene sovrascritto

```c
int x=1;
printf('Questo è il valore di x %d \n', x);
x=3;
printf('Questo è il nuovo valore di x %d \n', x);
```

Le variabili devono essere un identificatore valido che va dichiarato (insieme al tipo) prima di poterla utilizzare (il nome dovrebbe essere descrittivo del contenuto)
### Dove dichiarare le variabili
- **Globali** → fuori dalle funzioni
- **Parametri** → nell’header di una funzione
- **Locali** → all’interno del blocco di codice di una funzione (o all’inizio della funzione o poco prima del suo primo uso)

#### Variabili dichiarate all’inizio della funzione
**Vantaggi**
- Contesto storico → il linguaggio C è stato sviluppato all'inizio degli anni '70, quando molti linguaggi di programmazione seguivano la pratica di dichiarare tutte le variabili all'inizio di un blocco. Questa convenzione era influenzata da linguaggi precedenti come ALGOL.
- Allocazione della memoria → dichiarare le variabili all'inizio di un blocco permette al compilatore di allocare la memoria per tutte le variabili in una sola volta, migliorando le prestazioni. La disposizione della memoria è più semplice e può essere ottimizzata più facilmente.
- Visibilità dello scope → dichiarando le variabili all'inizio di un blocco, il loro scope è chiaro e prevedibile. Tutte le variabili dichiarate in un blocco sono visibili per l'intera durata del blocco, il che aiuta nella leggibilità e nella comprensione del flusso del codice.
- Prevenzione degli errori → la dichiarazione anticipata può aiutare a prevenire errori legati all'uso delle variabili. Se una variabile viene utilizzata prima di essere dichiarata, il compilatore genererà un errore, rendendo più facile individuare gli errori.
- Chiarezza del codice → dichiarare le variabili all'inizio può rendere il codice più leggibile e facile da mantenere. Permette agli sviluppatori di vedere tutte le variabili che verranno utilizzate in una funzione o in un blocco fin dall'inizio.

**Svantaggi**
- Potrebbe essere meno ovvio dove queste variabili sono usate per la prima volta e dove sono inizializzate
- Promuove il riuso di variabili per diverse necessità che non è una buona abitudine

#### Variabili dichiarate vicino al punto di primo uso
**Vantaggi**
- Ridurre al minimo la loro “durata”
- Sfavorire il riutilizzo

**Svantaggi**
- Non mostrare tutte le variabili locali utilizzate in un unico posto.

### Dichiarazione di variabili e costanti
```c
optional_modifier data_type name_list
```

`optional_modifier` indica dei modificatori applicati al tipo di dato (es. `signed`, `unsigned`, `short`, `long`, `const`)

`data_type` specifica il tipo di valore che permette al compilatore di sapere quali sono le operazioni consentite sul tipo di dato e come deve essere rappresentato in memoria

`name_list` indica la lista di nomi delle variabili

>[!example]
>```c
>int x, y, z; short int a; long int b;
>const float TAX_RATE=31.5;
>```

---
## Tipi per i numeri
![[Pasted image 20250324201607.png|center|400]]


| Type             | Storage size | Value range                                        |
| ---------------- | ------------ | -------------------------------------------------- |
| `char`           | 1 byte       | -128 a 127 o 0 a 255                               |
| `unsigned char`  | 1 byte       | 0 a 255                                            |
| `signed char`    | 1 byte       | -128 a 127                                         |
| `int`            | 2 o 4 bytes  | -32.768 to 32.767 o -2.147.483.648 a 2.147.483.647 |
| `unsigned int`   | 2 o 4 bytes  | 0 a 65.535 o 0 a 4.294.967.295                     |
| `short`          | 2 bytes      | -32.768 a 32.767                                   |
| `unsigned short` |              |                                                    |
