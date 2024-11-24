---
Created: 2024-11-18
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
Il SO è responsabile dell’assegnamento di blocchi a file, ma ci sono due problemi correlati (influenzati l’un l’altro):
- occorre allocare spazio per i file, e mantenerne traccia una volta allocato
- occorre tenere traccia dello spazio allocabile

I file si allocano in “porzioni” o “blocchi” (non byte a byte per motivi di efficienza) la cui dimensione minima è il settore del disco, ma di solito ogni porzione o blocco è una sequenza contigua di settori

---
## Allocazione di spazio per i file
Per l’allocazione di spazio per i file ci sono vari problemi da affontare:
- decidere se fare **preallocazione** o **allocazione dinamica**
- decidere se lo spazio deve essere diviso in parti di dimensione fissa (**blocco**) o di dimensione dinamica (**porzione**)
- quale deve essere il metodo di allocazione: **contiguo**, **concatenato** o **indicizzato**
- gestione della *file allocation table* (per ogni file, serve a mantenere le informazioni su dove si trovano le porzioni che lo compongono sul disco)

---
## Preallocazione vs. allocazione dinamica
Quando si parla di **preallocazione** si intende che la dimensione massima di un file viene dichiarata a tempo di creazione, che risulta facilmente stimabile in alcune applicazioni (es. risultato di compilazioni, file che forniscono sommari sui dati) ma difficile su molte altre infatti utenti ed applicazioni tendono a sovrastimare la dimensione risultando in uno spreco di spazio su disco, a fronte di un modesto risparmo di computazione.

Per questo motivo viene quasi sempre preferita l’**allocazione dinamica**, tramite la quale la dimensione del file viene aggiutasta in base alle call `append` (aggiungere informazioni) o `truncate` (rimuovere informazioni)

---
## Dimensioni delle porzioni
Per decidere la dimensione delle porzioni si hanno due possibilità agli estremi:
- si alloca una **porzione larga a sufficienza per l’intero file** → efficiente per il processo che vuole creare il file (viene allocata della memoria contigua)
- si alloca **un blocco alla volta** → efficiente per il SO, che deve gestire tanti file, ma ciascun blocco è una sequenza di $n$ settori contigui con $n$ fisso e piccolo (spesso $n=1$)

Si cerca dunque un punto di incontro (*trade-off*) tra efficienza del singolo file ed efficienza del sistema.
Sarebbe ottimo, per le prestazioni di accesso al file, fare porzioni contigue, ma inefficiente per l’SO; invece porzioni piccole vuol dire grandi tabelle di allocazione, e quindi grande overhead, ma vuol dire anche maggior facilità nel riuso dei blocchi.
Sarebbe inoltre da evitare di fare porzioni fisse grandi in quanto porterebbe a frammentazione interna, ma rimane possibile la frammentazione esterna in quanto i file possono essere cancellati

Alla fine risultano quindi rimanere due possibilità (valide sia per preallocazione che per allocazione dinamica):
- **porzioni grandi e di dimensione variabile**
	- ogni singola allocazione è contigua
	- tabella di allocazione abbastanza contenuta
	- complicata la gestione dello spazio libero: servono algoritmi ad hoc
- **porzioni fisse e piccole**
	- tipicamente, 1 blocco per una porzione
	- molto meno contiguo del precedente
	- spazio libero: basta guardare una tabella di bit

Con la preallocazione viene naturale utilizzare porzioni grandi e di dimensione variabile. Infatti con questa combinazione non è necessaria la tabella di allocazione dato che per ogni file basta l’inizio e la lunghezza (ogni file è un’unica porzione) e come per il partizionamento della RAM si parla di best fit, first fit, next fit (ma qui non c’è un vincitore troppe variabili ed è molto inefficiente per lo spazio libero in quanto necessita periodica compattazione che è molto più onerosa rispetto a quella per la RAM).

---
## Come allocare spazio per i file
Per allocare spazio per i file si utilizzano tre metodi:
- **contiguo**
- **concatenato**
- **indicizzato**

 Per ciascuno di questi tre metodi ci sono delle caratteristiche
 ![[Pasted image 20241118210718.png]]
### Allocazione contigua
Con l’allocazione contigua un insieme di blocchi viene allocato per il file quando quest’ultimo viene creato. Dunque la preallocazione risulta necessaria, occorre infatti sapere quando lungo, al massimo sarà il file, altrimenti, se un file può crescere oltre il limite massimo potrebbe incontrare blocchi già occupati

In questo modo risulta necessaria una sola entry nella tabella di allocazione dei file (blocco di partenza e lunghezza del file).
Tipicamente ci sta frammentazione esterna, con conseguente necessità di compattazione

![[Pasted image 20241118213331.png|500]]

#### Compattazione
![[Pasted image 20241118213445.png|500]]

### Allocazione concatenata
Con l’allocazione concatenata viene allocato un blocco alla volta che ha un **puntatore al prossimo blocco** (la prima parte del blocco sono dati del file, l’ultima, piccola, parte del blocco è il puntatore)

Risulta quindi necessaria **una sola entry** nella tabella di allocazione dei file contenente il blocco di partenza e la lunghezza del file (lunghezza anche calcolabile, ma è comodo avere già il valore calcolato)
In questo modo **non si ha frammentazione esterna** (la frammentazione interna è trascurabile, in quanto molto piccola)
Con questo tipo di allocazione non risulta essere un problema accedere ad un file in modo sequenziale, ma se serve un blocco che si trova $b$ blocchi dopo quello iniziale, occorre scorrere tutta la lista. Per risolvere questo problema si ricorre al **consolidamento**, analogo alla compattazione, per mettere i blocchi di un file contigui e migliorare l’accesso sequenziale

![[Pasted image 20241118220632.png|500]]

#### Consolidamento
![[Pasted image 20241118220703.png|500]]

### Allocazione indicizzata
L’allocazione indicizzata è quella che, in seguito ad alcune modifiche, viene utilizzata nei computer odierni. Risulta essere una via di mezzo tra l’allocazione contigua e quella concatenata, risolvendo quasi tutti i problemi che le precedenti avevano.

