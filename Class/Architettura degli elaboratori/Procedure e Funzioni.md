---
Created: 2024-03-21
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [[#Introduction|Introduction]]
>- [[#Ingredienti|Ingredienti]]

---
## Introduction
Una **funzione** è un frammento di codice che riceve degli argomenti e calcola un risultato (utile per rendere il codice riusabile e modulare)

![[Screenshot 2024-03-19 alle 13.20.59.png|500]]
Una funzione (o procedura) in assembly:
- ha un indirizzo di partenza
- riceve uno o più argomenti
- svolge un calcolo
- ritorna il risultato
- continua la sua esecuzione dall’istruzione seguente a quella che l’ha chiamata

## Ingredienti
Gli ingredienti principali per la creazione di funzioni in Assembly sono i **salti incondizionati**. In particolare delle istruzioni:
- `jal etichetta` → quest’istruzione, oltre che fare un jump all’etichetta (della funzione) indicata, salverà nel registro `$ra` l’indirizzo del Program Counter da cui è stato chiamato il jump
- `jr $ra` → questa istruzione viene eseguita al termine del corpo della funzione, così facendo infatti tornerà all’indirizzo del Program Counter successivo a quello da cui ho chiamato la funzione

Per convenzione vengono utilizzati i registri `$a0,$a1,$a2,$a3` per passare valori in input alla funzione, mentre `$v0,$v1` per restituire valori dalla funzioni

>[!info] Convenzioni
>- `$t0,$t1...` → possono cambiare tra una chiamata e l’altra (temporary)
>- `$s0,$s1...` → non cambiano tra una chiamata e l’altra (saved) 

