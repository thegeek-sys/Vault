---
Created: 2025-03-17
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
In Linux le due entità fondamentali sono:
- **file** → descrivono/rappresentano le risorse
- **processi** → permettono di elaborare dati e usare le risorse

Un file eseguibile, in esecuzione è chiamato **processo**. Per lanciare un processo bisogna eseguire il file corrispondente

>[!example]
>Esempi di processi sono quelli creati eseguendo i comandi delle lezioni precedenti (es. `dd`, `ls`, `cat`, …)

Però non tutti i comandi shell creano dei processi. Ad esempio `echo` e `cd` vengono esguito all’interno del processo di shell

Un file eseguibile più essere eseguito più volte dando vita ad un nuovo processo ogni volta e non è necessario attendere il termine della prima esecuzione per avviare la seconda (Linux è multi-processo)

### Ridirezione dell’output
I simboli `>` e `<` possono essere utilizzati per redirigere l’output di un comando su di un file
Ad esempio:
- `ls > dirlist` → output di `ls` redirezionato in `dirlist`
- `ls > dirlist 2 > &1` →