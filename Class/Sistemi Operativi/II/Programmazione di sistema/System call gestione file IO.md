---
Created: 2025-04-28
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## Operare sui file

>[!info]
>Per file si intende una astrazione che descrive ogni risorsa (es. file su disco, dispositivi, socket, fifo)

Sequenza tipica di azione su un file:
1. apertura file mediante `open` → viene creato un file descriptor
2. operazione su file mediante syscall → es. `write`, `read`
3. chiusura del file mediante `close`

Quando un processo termina, chiude automaticamente tutti i file correntemente aperti, tuttavia, è sempre consigliabile e buona regola di programmazione chiudere esplicitamente un file invocando la `close`

---
## File descriptor
Un **file descriptor** è il riferimento ad un file aperto ed è rappresentato da un intero piccolo, generato sequenzialmente a partire dal valore $0$.

![[Pasted image 20250428214601.png|400]]

Per default, ogni processo, ha associato i seguenti file descriptor:
- $0$ → `stdin`
- $1$ → `stdout`
- $2$ → `stderr`

Quando un file viene chiuso, il suo file descriptor viene liberato e può essere riutilizzato. E’ inoltre possibile aprire uno stesso file ed ottenere file descriptor diversi che puntano allo stesso file

### File flags associati a file descriptor
Si differenziano due tipi di flags:
- **file status flags** → associati allo stato del file e condivisi tra tutti i file descriptor ottenuti per duplicazione da un unico file descriptor
- **file descriptor flags** →
	- associati al file descriptor (non al file aperto)
	- indipendenti dal contenuto o status del file (file descriptor diversi che fanno riferimento ad uno stesso file hanno ognuno il proprio insieme di descriptor flags)
	- descrivono proprietà e comportamento delle operazioni effettuate sul file
	- alcuni definiti solo per alcuni tipi di file speciali (es. fifo, socket)

#### Rappresentazione dei flag
I flag sono rappresentati con maschere di bit