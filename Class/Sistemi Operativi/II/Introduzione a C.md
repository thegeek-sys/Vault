---
Created: 2025-03-24
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Ambiente di sviluppo in C
![[Pasted image 20250324183949.png]]
`gcc` (*GNU Compiler Coollection*) include il compilatore C che svolge l’attività di pre-processamento, compilazione e linking e che produce **object code**, memorizzato in un file `.o`

### Ambiente di esecuzione in C
![[Pasted image 20250324184141.png]]
Load e execute vengono eseguite dal sistema operativo

---
## Differenze con Java (e Python)
La compilazione di un programma in Java mediante javac non produce codice eseguibile ma codice interpretabile dalla JVM contenuto nel `.class` (bytecode). Infatti quando voglio eseguire un `.class` lo do in pasto alla JVM (il vero processo in esecuzione è la JVM che esegue il `.class` interpretandolo)

La “compilazione” di un programma in C mediante **gcc** (fasi 2-4) produce un “object code” che è un file eseguibile. Eseguendo il file quindi viene creato un processo indipendente dal gcc che eventualmente posso eseguire su altri sistemi senza necessità di ricompilarlo (lo stesso vale per un programma scritto in C++), al contrario il file `.class` dovrò sempre darlo in pasto ad una JVM

---
## Struttura di un programma C
Tipicamente un programma C è strutturato in due parti:
- **Main function** (compulsory) → può essere semplicemente il punto da cui vengono invocate tutte le funzioni che compongono il programma o può contenere una logica complessa (può anche essere l’ultimo blocco di codice del programma)
- **Function** → un blocco di codice che esegue uno specifico compito identificato da un nome univoco

>[!info]
>Main function e functions possono  risiedere nello stesso file `.c` o in file diversi `.c`

### Functions
Ogni funzione consiste di un header e di un basic block

```
<return-type> fn-name (parameter-list)
	basic block
```

Basic block
```
{
	declaration of variables
	executable statements
}
```

### Return statement
```
return expression
```
Imposta il valore di ritorno di una funzione al valore di `expression` e ritorna il valore al caller/invoker

```c
int main()    // header
{             // beginning of basic block
              // ...
	return 0; // program ending successfully
}             // end of basic block
```

`expresion` può essere una costante, una variabile, un’espressione logico-aritmetica, una funzione

---
## Per compilare ed eseguire
```
gcc -Wall prog-name.c
```
In questo modo vengono stampati tutti i messaggi di warning (se presenti)

```
gcc -Wall prog-name.c -lm
```
Il flag `-lm` va specificato se si includono le librerie matematiche `<math.h>`, ad esempio per usare funzioni come `sin`, `cos`, `log`, `ln`, ecc.

Il risultato si trova in un file eseguibile `a.out`
Per specificare il nome dell’output
```
gcc -Wall prog-name.c -o executable-name.o
```

### Precompilazione, compilazione e linking
Per fare solo la precompilazione (o preprocessamento)
```bash
cpp helloworld.c > precompilato.c
```
Esegue tutte le direttive del compilatore ed elimina i commenti

Solo compilazione (di un precompilato)
```bash
gcc -c precompilato.c -o test.o
```
In questo modo `gcc` controlla che la sia sintassi sia corrette e per ogni chiamata a funzione, controlla che venga rispettato il corrispettivo header (che quindi deve esistere al momento della chiamata) e infine crea effettivamente del codice macchina, ma solo per il contenuto delle funzioni (ogni chiamata a funzione ha una destinazione simbolica)

