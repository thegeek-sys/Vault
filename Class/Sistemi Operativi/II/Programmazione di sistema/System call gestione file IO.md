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
I flag sono rappresentati con maschere di bit. Per poter gestire i flag si usano dei bit (ciascun bit rappresenta un flag). Quindi:
- `1` → flag attivo
- `0` → flag non attivo

Per ogni flag esiste una maschera (valore numerico dove un certo bit è $1$ e tutti gli altri $0$). Ad esempio `O_RDONLY` è una maschera
Se si vogliono attivare più flag insieme è possibile fare un ora bitwise

```
MACRO1 = 01000000;
MACRO2 = 00010000;
MACRO1|MACRO2 = 01010000;
```

#### File status flags
Esistono tre categorie di file status flags:
- *modalità di accesso* → `read`, `write`, `read&write` che sono specificati nella `open` e non possono essere modificati una volta aperto il file
- *di apertura* → definiscono il comportamento della open e non vengono mantenuti
- *modalità operative* → definiscono il comportamento delle operazioni `read` e `write` e sono specificati nella open ma possono essere modificati anche dopo l’apertura del file

---
## System call principali
- `open`
- `chown`
- `read`
- `chmod`
- `write`
- `stat`
- `lseek`
- `select`
- `close`
- `ioctl`
- `unlink`
- `fnctl`
- `symlink`
- `rename`
- `rmdir`
- `chdir`

