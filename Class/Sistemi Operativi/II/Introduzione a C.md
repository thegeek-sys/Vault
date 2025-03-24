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

