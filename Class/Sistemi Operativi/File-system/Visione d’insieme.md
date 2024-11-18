---
Created: 2024-11-18
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## I file(s)
I files sono l’elemento principale per la maggior parte delle applicazioni, nella maggior parte dei casi infatti un file è proprio l’input dell’applicazione, ma altrettanto spesso anche l’output è un file.
In particolar modo i file sono importanti in quanto questi sopravvivono anche dopo la morte dei processi (a differenza della RAM del processo che invece viene sovrascritta)

Dunque si può dire che il file system è una delle parti del sistema operativo che sono più importanti per l’utente

Queste sono le proprietà che sono richieste per rendere i file più usufruibili possibile dall’utente:
- esistenza a lungo termine
- condivisibilità con altri processi (tramite nomi simbolici)
- strutturabilità (directory gerarchiche)

### Gestione dei file
I file sono gestiti da un insieme di programmi e librerie di utilità all’interno del kernel. I programmi sono comunemente conosciuti come il *File (Management) System* e ovviamente vengono eseguiti in kernel mode. Le librerie, invece, vengono invocate come system call (sempre in kernel mode)

Questi programmi e librerie hanno a che fare con la memoria secondaria (dischi, chiavi USB, …). Particolarità sta nel fatto che in Linux, selezionate porzioni di RAM, possono essere gestite come file. Forniscono infatti un’astrazione sotto forma di operazioni tipiche (creazione, modifica, etc.) e per ogni file vengono mantenuti degli attributi (o metadati), come proprietario, data creazione, etc.

### Operazioni tipiche sui file
Le operazioni standard eseguibili su un file sono:
- Creazione (con annessa scelta del nome, al momento della creazione il file è tipicamente vuoto)
- Cancellazione
- Apertura → necessaria per poter leggere e scrivere
- Lettura → solo sui file aperti (e non chiusi nel frattempo)
- Scrittura → solo sui file aperti (e non chiusi nel frattempo)
- Chiusura → necessaria per le performance

---
## Terminologia
Introduciamo delle terminologie comuni (seppur datata), introduciamo quindi:
- **Campo** (field)
- **Record**
- **File**
- **Database**

### Campi e Record
**Campi**
I campi sono i valori di base, contengono infatti dei valori singoli. Sono caratterizzati da una lunghezza e dal tipo di dato (con demarcazioni). Esempio tipico: carattere ASCII

**Record**
Mettendo insieme campi diversi (ma correlati) ottengo un record ed ognuno di essi viene trattato come un’unità. Esempio tipico: un impiegato è caratterizzato dal record nome, cognome, matricola, stipendio

### File e Database
**File**
Mettendo insieme record ottengo un file (nei SO generici moderni, ogni record è un solo campo con un byte, stream di byte). Ognuno è trattato come un’unità con nome proprio e possono implementare meccanismi di controllo dell’accesso (alcuni utenti possono accedere ad alcuni file, altri ad altri)

**Database**
Collezioni di dati correlati formano un database. Questi mantengono anche relazioni tra gli elementi memorizzati e sono realizzati con uno o più file. Sono inoltre gestiti dai DBMS, che sono tipicamente processi di un sistema operativo

---
## Sistemi per 