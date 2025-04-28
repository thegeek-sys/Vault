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

---
## Nascita e morte
![[Pasted image 20250429011745.png]]

Ogni processo nella gerarchia fa riferimento al processo padre, ovvero quando nasce eredita codice e parte dello stato dal processo padre invece quando termina/muore ritorna l’exit status al processo padre (nel caso in cui il processo padre termini/muoia prima del processo figlio, quest’ultimo è adottato da `init`)

---
## Zombie
![[Pasted image 20250429011931.png|300]]

Un processo zombie è un processo terminato, ma il cui PCB è mantenuto nella Process Table dal kernel per dare modo al processo padre di leggere l’exit status di tale processo. Se il padre di un processo muore/termina il processo rimane in stato zombie

---
## System call controllo processi
![[Pasted image 20250429012027.png]]
![[Pasted image 20250429012044.png]]

Richiedono:
```c
#include <sys/types.h>
#include <unistd.h>
```

