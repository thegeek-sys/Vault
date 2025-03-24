---
Created: 2025-03-24
Class: "[[Sistemi Operativi]]"
Programming language: "[[C]]"
Related:
---
---
## Index
- [[#Ambiente di sviluppo in C|Ambiente di sviluppo in C]]
	- [[#Ambiente di sviluppo in C#Ambiente di esecuzione in C|Ambiente di esecuzione in C]]
- [[#Differenze con Java (e Python)|Differenze con Java (e Python)]]
- [[#Struttura di un programma C|Struttura di un programma C]]
	- [[#Struttura di un programma C#Functions|Functions]]
	- [[#Struttura di un programma C#Return statement|Return statement]]
- [[#Per compilare ed eseguire|Per compilare ed eseguire]]
	- [[#Per compilare ed eseguire#Precompilazione, compilazione e linking|Precompilazione, compilazione e linking]]
- [[#Direttive al processore $\verb|#|$|Direttive al processore $\verb|#|$]]
- [[#Input e Output|Input e Output]]
	- [[#Input e Output#Output|Output]]
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

Solo linking
```bash
gcc file.o
```
Risolve tutte le chiamate a funzione: adesso, per ogni funzione chiamata non basta più l’header, ci deve essere anche l’implementazione (blocco di istruzioni). L’implementazione più essere o data dal programmatore o fornita da librerie esterne. L’inclusione delle librerie puo’ essere automatica o specificata dall’utente (ad esempio la libreria `libc.a` che contiene la `printf` e’ inclusa automaticamente)

---
## Direttive al processore $\verb|#|$
Con `#` si indicano le direttive al processore

`#include filename` dice di includere il contenuto di `filename` specificato al posto di `#`
Il file (usualmente  `.h`) è detto **header file**

>[!example]
>```c
>#include <stdio.h>
>```
>- `<>` → indicano che il file header è un file standard del C in `/usr/include`
>- `""` → indicano che il file header è dell’utente e si trova nella directory corrente o in un path specificato
>- `-I` → permette di specificare le directory in cui cercare gli header file

---
## Input e Output
L’**input** viene letto da tastiera o altro dispositivo e memorizzato in variabili, l’**output** invece p visualizzato a schermo o inviato ad un altro dispostivo (es. stampante) prendendo valori da variabili
L’ambiente run-time del C, quando un programma viene eseguito, apre 2 file: `stdin` e `stdout`

>[!hint] Tutte le funzioni essenziali per l’I/O sono nel file `stdio.h`

### Output
```c
printf("format string", value-list);
```
In questo caso `value-list` può contenere:
- sequenze di caratteri
- variabili
- costanti
- espressioni logico-matematiche

`printf` riceve valori, ma il C permette di manipolare anche indirizzi di memoria e passarli come input a funzioni (per stampare il contenuto di una locazione di memoria di cui si conosce l’indirizzo si usa `scanf`)

Vedremo che ci sono altre funzioni per gestire l’output

Nella `printf` possiamo controllare la spaziatura orizzontale e verticale, e l’output di caratteri speciali utilizzano le sequenze di escape `\`

| Escape Sequence | Name               | Description                                                                            |
| --------------- | ------------------ | -------------------------------------------------------------------------------------- |
| `\a`            | Alarm or Beep      | It is used to generate a bell sound in the C program.                                  |
| `\b`            | Backspace          | It is used to move the cursor one place backward.                                      |
| `\f`            | Form Feed          | It is used to move the cursor to the start of the next logical page.                   |
| `\n`            | New Line           | It moves the cursor to the start of the next line.                                     |
| `\r`            | Carriage Return    | It moves the cursor to the start of the current line.                                  |
| `\t`            | Horizontal Tab     | It inserts some whitespace to the left of the cursor and moves the cursor accordingly. |
| `\v`            | Vertical Tab       | It is used to insert vertical space.                                                   |
| `\\`            | Backlash           | Use to insert backslash character.                                                     |
| `\’`            | Single Quote       | It is used to display a single quotation mark.                                         |
| `\”`            | Double Quote       | It is used to display double quotation marks.                                          |
| `\?`            | Question Mark      | It is used to display a question mark.                                                 |
| `\ooo`          | Octal Number       | It is used to represent an octal number.                                               |
| `\xhh`          | Hexadecimal Number | It represents the hexadecimal number.                                                  |
| `\0`            | NULL               | It represents the NULL character.                                                      |
