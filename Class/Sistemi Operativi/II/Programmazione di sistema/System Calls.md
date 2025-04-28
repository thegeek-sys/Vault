---
Created: 2025-04-29
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## La creazione $\verb|fork()|$
`Init` è il processo $0$ (`pid=1`) padre di tutti i processi del sistema in esecuzione. Da esso vengono creati tutti i processi mediante system call `fork()`

La creazione consiste nella duplicazione del processo per creare relazioni $\text{padre}\to \text{figlio}$
Tramite il comando `pstree 1` è possibile visualizzare l’albero (genealogico) dei processi

