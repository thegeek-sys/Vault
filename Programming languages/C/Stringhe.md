---
Created: 2025-04-06
Class: "[[Sistemi Operativi]]"
Programming language: "[[C]]"
Related:
---
---
## Introduction
Le stringhe in C sono degli **array di caratteri**, ma array particolari.
Infatti ogni elemento dell’array è un carattere della stringa e l’ultimo elemento contiene il carattere di fine stringa.

>[!example]
>`Hello` è memorizzato come `{H,e,l,l,o,\0}` dove `\0` è detto `NULL`
>![[Pasted image 20250406181444.png]]

---
## Inizializzazione

```c
char s[10] = 'Lezione 9';
```
![[Pasted image 20250406181600.png]]

```c
char s[10] = 'L9 4apr';
```
![[Pasted image 20250406181632.png]]

>[!hint] Carattere di fine stringa aggiunto automaticamente
>```c
char r[10]={‘L‘,’9’,’ ’,’4’,’a’,’p’,’r’};
>```
>
>In questo caso invece `\0` non è inserito automaticamente, quindi non è una stringa

---
## Lunghezza, copia, confronto e concatenazione
All’interno della libreria `string.h` sono presenti diverse funzioni comode per poter gestire le stringhe

### Lunghezza
Per calcolare la lunghezza di una stringa si usa il comando:
```c
size_t strlen(const char *s);
```
Che prende in input una stringa (insieme di `char`) e restituisce un intero

### Copia
Per creare una copia di una stringa si usa il comando:
```c
char *strcpy(char *dest, const char *src);

// copia al più n byte
char *strncpy(char *dest, const char *src, size_t n);
```
I due comandi ritornando il puntatore alla stringa di destinazione `dest`

Si può utilizzare `strcpy` anche per assegnare un valore ad una variabile (previa inizializzazione)

>[!example]
>```c
strcpy(s, “Lezione 9“);
>```

### Confronto
Per confrontare due stringhe si usa il comando:
```c
int strcmp(const char *s1, const char *s2);

// confronta i primi n byte di s1 e s2
int strcmp(const char *s1, const char *s2, size_t n);
```
La funzione `strcmp()` restituisce un valore che indica la relazione tra le due stringhe, come segue:

| Valore | Significato |
| ------ | ----------- |
| $<0$   | `s1<s2`     |
| $0$    | `s1==s2`    |
| $>0$   | `s1>s2`     |

### Concatenazione
Per concatenare due stringhe si usa:
```c
char *strcat(char *dest, const char *src);

// concatenati al più i primi n byte
char *strncat(char *dest, const char *src, size_t n);
```
La funzione restituisce un puntatore alla stringa di destinazione `dest`