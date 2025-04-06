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
char *strcpy(char *dest, const char *src)
```