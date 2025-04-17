---
Created: 2025-04-07
Programming language: "[[C]]"
Class: "[[Sistemi Operativi]]"
Related:
---
---
## Tipi di file
Esistono tre tipi di file:
- **testo**
- **binari**
- **buffer** → area di memoria temporanea che mantiene i dati che vengono trasferiti dal disco alla memoria o viceversa

### File di testo
Tutte le funzioni per la loro gestione sono contenute in `stdio.h` e tutti i file di testo terminano con `EOF` (end-of-file)

---
## Steps per usare i file
Dobbiamo seguire questi steps per poter usare i file in C:
- dichiarare un file pointer
- aprire il file
	- viene creata una struttura dati con le info necessarie per gestire il file
	- viene allocato l’area di buffer
	- collega il file pointer con la struttura
- usare le funzioni di I/O
- chiudere il file
	- scrivere il contenuto buffer nel file (se necessario)
	- libera la memoria associata al file pointer

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
	FILE *fp; // dichiarazione file pointer
	
	// apertura un file
	//FILE *fopen(const char *pathname, const char *mode);
		// mode = "r",  "w", "a", "r+", "w+"
		// w e a creano il file se non esiste
		// r+ w+ aprono entrambe in R/W mode
	fp=fopen("file.txt", "r");
	if (fp==NULL) {
		printf("errore in apertura\n");
		exit(1);
	}
	
	// chiusura file
	if (fclose(fp)==0) printf("chiusura corretta");
	else printf("errore  in chiusura");
}
```

---
## Input/output
```c
int fscanf(FILE *stream, const char *format, ...);
int fprintf(FILE *stream, const char *format, ...);
char *fgets(char *s, int size, FILE *stream);
int fputs(const char *s, FILE *stream);

// restituisce valore diverso da 0 se e solo trova EOR; altrimenti 0
int feof(FILE *stream);2
// posizione il puntatore all'inizio del file
void rewind(FILE *stream);
```
