---
Created: 2024-04-28
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Nel caso in cui la CU generi **segnali errati** dobbiamo individuare:
- quale combinazione di segnali viene generata
- quali istruzioni vengono influenzate dalle nuove combinazioni e cosa fanno
E solo a questo punto possiamo cercare di scrivere un breve programma che evidenzia se la CPU sia malfunzionante o meno

![[Screenshot 2024-04-29 alle 16.12.43.png]]

---
## Esempio: RegWrite ← Branch
Si ha il dubbio che, per difetto di progettazione della CU (es. cortocircuito) il segnale `RegWrite` sia determinato dal segnale `Branch`

Si assume che:
- `MemToReg=1` → solo per la `lw` ed altrimenti valga 0 (mai X)
- `RegDest=1` → solo per le istruzioni di tipo R ed altrimenti valga 0 (mai X)

1. Individuare quali istruzioni ne sono affette e perché
2. Scrivere un breve programma che scriva in `$s0` un valore che distingua se la CPU sia correttamente funzionante (`$s0=1`), o malfunzionante (`$s0=0`)

### 1)
![[Screenshot 2024-04-29 alle 16.18.09.png]]

Le istruzioni affette da questo guasto sono:
- tutte le istruzioni che modificano un registro (tipo R e `lw`) → lo lasceranno invariato
- `branch` → oltre ad effettuare il salto modificherà uno dei registri
	- `rt` sarà sovrascritto poiché `RegDest=0`
	- il valore scritto sarà il risultato della differenza usata per confrontare i due operandi perché assumiamo `MemToReg=0`
Le istruzioni sw e jump, invece, funzioneranno correttamente (perché hanno RegWrite a 0 comunque)

![[Immagine 29-04-24 - 16.58 2.jpg]]

### 2)
Per individuare se la CPU sia malfunzionante creiamo un programma che lasci il valore 0 nel registro $s0 se lo è, e scriva 1 se funziona correttamente, tenendo conto che non possiamo scrivere in un registro dato che `RegWrite=0`

Quindi provo a caricare su `$s1` il valore `1` e verificare se ciò è avvenuto oppure no
```asm
li $s0,1
```

---
## Esempio: MemWrite ← not(RegWrite)
Si ha il dubbio che in alcune CPU MIPS la Control Unit sia rotta, producendo il segnale di controllo MemWrite attivo se e solo se NON è attivo il segnale RegWrite.

Si assume che:
- `MemToReg=1` → solo per la `lw` ed altrimenti valga 0 (mai X)
- `RegDest=1` → solo per le istruzioni di tipo R ed altrimenti valga 0 (mai X)

 1. Indicare quali delle istruzioni funzioneranno male e perché.
2. Scrivere un breve programma assembly MIPS che termini scrivendo nel registro `$s0` il valore 1 se il processore è guasto, altrimenti vi scriva 0.

### 1)
![[Screenshot 2024-04-29 alle 16.54.24.png]]

Quindi sono danneggiate le istruzioni che hanno i due segnali entrambi a 0 o entrambi a 1
- `lw` → funziona correttamente (1, 0)
- `sw` → funziona correttamente (0, 1)
- `beq` → salta correttamente **ma inoltre scrive in memoria** (0, 1) – invece che (0, 0)
- `j` → salta correttamente **ma inoltre scrive in memoria** (0, 1) – invece che (0, 0)
- tipo R → funzionano correttamente (1, 0)
