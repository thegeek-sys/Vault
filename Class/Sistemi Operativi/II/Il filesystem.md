---
Created: 2025-03-07
Class: 
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Path|Path]]
	- [[#Path#Current working directory $ verb cwd $|Current working directory cwd]]
- [[#Contenuto di una directory|Contenuto di una directory]]
- [[#Creazione di directory e file|Creazione di directory e file]]
- [[#root directory|root directory]]
	- [[#root directory#mouting|mouting]]
- [[#Partizioni e mounting|Partizioni e mounting]]
- [[#Tipi di filesystem|Tipi di filesystem]]
	- [[#Tipi di filesystem#$ verb mount $|mount]]
- [[#File $ verb passwd $ e $ verb groups $|File passwd e group]]
- [[#I file|I file]]
	- [[#I file#Tabella inode|Tabella inode]]
	- [[#I file#Visualizzare le informazioni contenute nell’inode di un file|Visualizzare le informazioni contenute nell’inode di un file]]
- [[#Permessi di accesso ai file|Permessi di accesso ai file]]
	- [[#Permessi di accesso ai file#Permessi speciali|Permessi speciali]]
		- [[#Permessi speciali#Sticky bit (t)|Sticky bit (t)]]
		- [[#Permessi speciali#Setuid bit (s)|Setuid bit (s)]]
		- [[#Permessi speciali#Setgid bit (s)|Setgid bit (s)]]
	- [[#Permessi di accesso ai file#Visualizzare attributi di accesso|Visualizzare attributi di accesso]]
	- [[#Permessi di accesso ai file#Visualizzare permessi speciali|Visualizzare permessi speciali]]
	- [[#Permessi di accesso ai file#Settare i permessi|Settare i permessi]]
		- [[#Settare i permessi#Formato ottale|Formato ottale]]
		- [[#Settare i permessi#Modalità simbolica|Modalità simbolica]]
- [[#Comandi|Comandi]]
	- [[#Comandi#$ verb umask [mode] $|umask]]
	- [[#Comandi#$ verb cp [-r] [-i] [-a] [-u] {file_src} file_dst $|cp]]
	- [[#Comandi#$ verb mv [-i] [-u] [-f] {file_src} file_dst $|mv]]
	- [[#Comandi#$ verb rm [-f] [-i] [-r] {file} $|rm]]
	- [[#Comandi#$ verb ln [-s] src [dest] $|ln]]
	- [[#Comandi#$ verb touch [-a] [-m] [-t timestamp] {file} $|touch]]
	- [[#Comandi#$ verb du [-c] [-s] [-a] [-h] [--exclude=PATTERN] [files] $|du]]
	- [[#Comandi#$ verb df [-h] [-l] [-i] [file] $|df]]
	- [[#Comandi#$ verb dd [opzioni] $|dd]]
	- [[#Comandi#$ verb mkfs [-t type fs-options] device $|mkfs]]
---
## Introduction
L’organizzazione di un area di memoria è basata sul concetto di file e di directory
Una directory contiene file e directory ed ha una struttura gerarchica ad albero (ma solo le directory possono avere figli)

I file regolari contengono sequenze di bit dell’area di memoria sulla quale c’è il filesystem (ASCII o binari), ma esistono file **non regolari** o speciali

Linux ha un solo filesystem principale che ha come directory radice `/` (root). Tutti i file e directory sono contenuti direttamente o indirettamente in `/`
In una stessa directory è vietato creare:
- 2 file con lo stesso nome
- 2 directory con lo stesso nome
- un file ed una directory con lo stesso nome

---
## Path
Ogni file o directory è raggiungibile dalla directory principale `/` mediante un **path assoluto**, ovvero una sequenza di directory separate da `/`

```bash
/home/utente1/dir1/dir11/dir112/file.pdf
```

>[!info]
>Un’eccezione di path assoluto riguarda l’utente corrente (`userX`), infatti `~` equivale a `/home/userX`

### Current working directory $\verb|cwd|$
Per conoscere la current working directory si usa il comando `pwd`. Per cambiare cwd usare il comando `cd [path]` dove `[path]` può essere assoluto o relativo (`cd` senza path ritorna alla home)
E’ possibile inoltre usare `..` (parent directory) e `.` (directory attuale)

---
## Contenuto di una directory
Il comando `ls` restituisce la lista dei file contenuti in una directory. Però in una directory ci sono dei file “nascosti” (tipicamente file di configurazione o sile usati a supporto di comandi e applicazioni, es. `.bash_history`); questi sono visibili attraverso `ls [-a | --all]`

Se vogliamo ricorsivamente visualizzare il contenuto delle sotto directory abbiamo l’opzione `[-R | --recursive]`

Si puo’ visualizzare l’albero delle directory con il comando
```bash
tree [-a] [-L maxdepth] [-d] [-x] [nomedir]
```

---
## Creazione di directory e file
Per creare directory:
```bash
mkdir [-p] nomedir
```
Crea la directory `nomedir` vuota. Se si vuole creare un intero path di directory (es. `./dir11/dir12/dir13`) è necessario l’opzione `-p`

Per creare un file:
```bash
touch nomefile
```
Crea il file `nomefile` vuoto

---
## root directory
Il filesystem root (`/`) contiene elementi eterogenei:
- disco interno solido o magnetico
- filesystem su disco esterno (es. usb)
- filesystem di rete
- filesystem virtuali (usati dal kernel per gestire risorse)
- filesystem in memoria principale

Tutto questo è possibile solo grazie al meccanismo del *mounting*

### mouting
Una qualsiasi directory $D$ dell’albero gerarchico può diventare punto di mount per un altro filesystem $F$ se e solo se la directory root di $F$ diventa accessibile da $D$.

Si hanno due possibilità per $D$:
- se $D$ è vuota, dopo il mount conterrà $F$
- se $D$ non è vuota, dopo il mount conterrà $F$ ma ciò non significa che i dati che vi erano dentro sono persi, saranno infatti **di nuovo disponibili dopo l’unmount** di $F$

---
## Partizioni e mounting
Un singolo disco può essere suddiviso in due o più partizioni.
Una partizione $A$ può contenere il sistema operativo e la partizione $B$ i dati degli utenti (home directory degli utenti). La partizione $A$ verrà montata su `/`, mentre la partizione $B$ verrà montata su `/home`

---
## Tipi di filesystem

| Nome     | Journal | Partiz (TB) | File (TB) | Nome file (bytes) |
| -------- | ------- | ----------- | --------- | ----------------- |
| ext2     | No      | 32          | 2         | 255               |
| ext3     | Si      | 32          | 2         | 255               |
| ext4     | Si      | 1000        | 16        | 255               |
| reiserFS | Si      | 16          | 8         | 4032              |

![[Pasted image 20250311114805.png]]

Ci sono anche file system non linux, ad esempio windows: NTFS, MSDOS, FAT32, FAT64. Di questi FAT (qualsiasi) e NTFS possono essere montati su un filesystem Linux
`mount` è il comando per montare un fsfilesystem e visualizzare il filesystem montati

### $\verb|mount|$
`mount` visualizza gli fs montati, ma ciò può essere fatto anche con:
- `cat /etc/mtab` → fs montati all’avvio del sistema
- `cat /etc/fstab` → fs montati al boot

>[!warning] Domanda di esame
>- Usare il comando mount o consultare il contenuto del file `/etc/mtab` per verdere I file system montati.
>- Con l’aiuto del `mount` cercare di capire le varie opzioni di mounting dei vari filesystems

---
## File $\verb|passwd|$ e $\verb|groups|$
Il file `/etc/passwd` contiene tutti gli utenti, mentre il file `/etc/groups` contiene tutti i gruppi. Rappresentano una delle filosofie di Linux, e poiché hanno una struttura ben definita e conosciuta dei programmi, questi ci possono direttamente interagire

![[Sicurezza#Utenze e gruppi]]

---
## I file
Ogni filesystem è rappresentato da una struttura dai **inode** ed è univocamente identificato da un **inode number**. La cancellazione di un file libera l’**inode number** che verrà riutilizzato quando necessario per un nuovo file

![[Pasted image 20250311120539.png|500]]

Principali attributi della struttura dati inode:
- **Type** → tipo di file (regular, block, fifo ...)
- **User ID** → id dell'utente proprietario del file
- **Group ID** → id del gruppo a cui e associato il file
- **Mode** → permessi (read, write, exec) di accesso per il proprietario, il gruppo e tutti gli altri
- **Size** → dimensione in byte del file
- **Timestamps** → *ctime* (inode changing time, cambiamento di un attributo), *mtime* (content modication time, solo scrittura), *atime* (content access time: solo lettura)
- **Link count** → numero di hard links
- **Data pointers** → puntatore alla lista dei blocchi che compongono il file; se si tratta di una directory, il contenuto su disco e costituito di una tabella con 2 colonne: nome del file/directory e suo inode number

### Tabella inode
All’inizio del disco è sempre presente una **tabella inode**

>[!example] Esempio per ext2
>![[Pasted image 20250311121231.png]]
>![[Pasted image 20250311121310.png|400]]
>![[Pasted image 20250311121337.png|400]]
>
>Come viene seguito un path:
>![[Pasted image 20250311121421.png]]

### Visualizzare le informazioni contenute nell’inode di un file
Tra le opzioni del comando `ls` vi è la possibilità di visualizzare l’inode number tramite il flag `-i`, e i diritti, user, group, date, size, time tramite l’opzione `-l`

![[Pasted image 20250312124306.png]]
Il numero dopo i diritti indica il numero di directory all’interno della directory (vengono contate anche `.` e `..`); per i file è ovviamente 1.
Totale invece rappresenta la dimensione della directory in blocchi su disco, ma non per il suo sottoalbero (normalmente un blocco ha dimensione tra $1\text{kB}$ e $4\text{kB}$)

L’opzione `-n` consente di visualizzare ID utente e ID gruppo invece dell’user esteso
Per vedere i timestamp si usa l’opzione `-l` ma in combinazione con:
- `-c` → **ctime** (change time, data di ultima modifica metadati del file)
- `-u` → **atime** (access time, data di ultimo accesso)
- senza niente → **mtime** (modification time, data di ultima modifica del contenuto del file)

La sintesi delle opzioni sopra elencate la si ha con il comando `stat filename`
![[Pasted image 20250315121755.png]]
Dove:
- `Size` → dimensione del file in byte
- `Blocks` → numero di blocchi allocati sul disco
- `IO Block` → dimensione del blocco I/O del filesystem

`stat -c %B filename` invece ci permette di stampare la dimensione in byte dei blocchi di I/O del filesystem su cui si trova il file (non la dimensione del file)

---
## Permessi di accesso ai file
Chi può fare cosa:
- Utente proprietario → solitamente chi crea il file/directory
- Gruppo proprietario → gruppo primario dell’utente proprietario (specificato in `/etc/passwd`)

Il proprietario è colui che definisce i permessi di accesso
![[Pasted image 20250315122439.png|300]]

### Permessi speciali
Esistono però anche dei permessi speciali che possono essere applicati a file e directory:
- sticky bit (t)
- stuid bit (s)
- setgit bit (s)

#### Sticky bit (t)
Questo viene applicato sulle directory per correggere il comportamento di `w+x`, permettendo la cancellazione di file se si hanno permessi di scrittura su essi.
Senza sticky bit infatti per cancellare un file in una directory, sarà sufficiente avere i diritti di scrittura sulla directory (non sono necessari permessi di scrittura sul file). Mentre con lo sticky bit sono **necessari sia i permessi di scrittura sulla directory che sul file**

#### Setuid bit (s)
Questo si usa solo per i file eseguibili. Garantisce che **quando sono eseguiti**, hanno i **privilegi del proprietario del file**, e non quelli dell’utente che lo esegue (quindi se il proprietario è root, viene eseguito con privilegi di root indipendentemente da chi lo ha eseguito)
Ad esempio il comando `passwd` ha il setuid, che permette ad un utente di moficare la propria password (nonostante il proprietario del comando sia `root`)

#### Setgid bit (s)
Risulta analogo di setuid ma per i gruppi (i privilegi sono quelli del gruppo che è proprietario del file eseguibile)
Può essere applicato anche ad una directory, e allora ogni file creato al suo interno ha il gruppo della directory, anziché quello primario di chi crea i file

### Visualizzare attributi di accesso
Tramite il comando `ls` o `stat`
![[Pasted image 20250315130553.png|400]]
### Visualizzare permessi speciali
Vengono visualizzati al posto del bit di esecuzione:
- il setuid nella terna utente (`user`)
- il setgit nella terna gruppo (`group`)
- lo sticky nella terna altro (`other`)

![[Pasted image 20250315124907.png]]

Se il permesso di esecuzione corrispondente c’è, allora la `s` o la `t` saranno minuscoli, altrimenti saranno maiuscoli

### Settare i permessi
```bash
chmod mode[, mode...] filename
```
Setta la modalità (diritti) di accesso a file o directory
Si può fare in due modi o tramite **formato ottale** o in **modalità simbolica** (lettere)
L’opzione `-R` può essere applicata solo da `root` ed applica il comando a tutte le sottodirectory
#### Formato ottale
Si usano 4 numeri tra 0 e 7, il primo indica setuid (4), setgid (2) e sticky (1), gli altri sono utente, gruppo ed altri
Si possono fornire 3 numeri se si assume setuid, setgid e sticky settati tutti a 0
#### Modalità simbolica
Per settare il mode usando dei simboli il formato è: `[ugoa][+-=][perms...]`, dove `perms` è:
- `zero`
- una o più lettere nell’insieme `{rxwXst}`
- una lettera nell’insieme `{ugo}`

---
## Comandi
### $\verb|umask [mode]|$
Setta la maschera dei file a `mode`, ovvero i diritti di accesso al file o alle directory nel momento della loro creazione.
La *umask* è rappresentata in numeri ottali (es. 022, 002, 077). Ogni cifra rappresenta i permessi da rimuovere per utente (u), gruppo (g) e altri (o). Per i file devo sottrarre a partire da `666` mentre per le directory a partire a `777`
Se ad esempio l’umask è `022`, allora i permessi per i file saranno `644` (`rw-r--r--`) mentre per le directory `755` (`rwxr-xr-x`)

### $\verb|cp [-r] [-i] [-a] [-u] {file_src} file_dst|$
- `-r` → per le directory (può perdere attributi e timestamp)
- `-i` → interactive per essere avvisati in caso di sovrascrizione
- `-u` → la sovrascrittura avviene solo se l’mtime del sorgente è più recente di quello della destinazione
- `-a` → copia ricorsiva che preserva attributi e timestamp

### $\verb|mv [-i] [-u] [-f] {file_src} file_dst|$
Sposta un file (o lo rinomina). Le opzioni hanno lo stesso significato di `cp`, invece `-f` indica force (è il default)

### $\verb|rm [-f] [-i] [-r] {file}|$
Stesso significato delle opzioni per rispetto ai precedenti

### $\verb|ln [-s] src [dest]|$
Questo comando serve per creare dei symbolic e hard link. Di default verranno creati degli hard link ma con `-s` diventano soft

### $\verb|touch [-a] [-m] [-t timestamp] {file}|$
Questo comando permette di creare dei file, ma anche di modificarne il timestamp:
- `-a` → modifica unicamente il tempo di accesso
- `-m` → modifica unicamente il tempo di modifica
- `-t timestamp` → usa il timestamp indicato invece del timestamp corrente seguendo questo formato `[[CC]YY]MMDDhhmm[.ss]`

### $\verb|du [-c] [-s] [-a] [-h] [--exclude=PATTERN] [files]|$
Questo comando stima lo spazio occupato del/dei file e/o directories dati in input. Vediamo ora le opzioni:
- `-c` → restituisce la somma del totale (di tutti i file passati in input)
- `-s` → non mostra eventuali sottodirectory o file contenuti
- `-a` → ricorsivo sulle directories in essa contenuta (valido solo per directory)
- `-h` → restituisce i risultati in formato leggibile (e.g., 1K 234M 2G)

### $\verb|df [-h] [-l] [-i] [file]|$
Restituisce un report sullo spazio libero. Vediamo le opzioni:
- `-h` → restituisce i risultati in formato leggibile (e.g., 1K 234M 2G)
- `-l` → mostra solo i filesystem locali (esclude quelli remoti come NFS o montaggi di rete)
- `-i` → restituisce le informazioni riguardanti gli inodi invece di quelle rispetto ai blocchi
- `file` → restituisce le informazioni rispetto al filesystem in cui è memorizzato il `file`

### $\verb|dd [opzioni]|$
Serve per copiare un file in modo elaborato e le opzioni sono una sequenza di `var=value`
Vediamo le opzioni:
- `bs=BYTES` → legge e scrive fino a `BYTES` bytes alla volta (default 512)
- `count=N` → copia solamente `N` blocchi di input
- `skip=N` → salta i primi `N` blocchi di input
- `seek=N` → salta i primi `N` blocchi di output
- `conv=CONVS` → converte il file come da comma-separated `CONVS` in input (es. `ucase`, `lcase`, `noerror`, …)
- `if=FILE` → leggi da `FILE` invece di `stdin` (`cat file | dd of=output_file bs=1M`); viene usato `/dev/zero` quando bisogna creare un file con soli zero
- `of=FILE` → scrivi su `FILE` invece che su `stdout`

Oltre che per le conversioni si usa per copiare file speciali che non possono essere copiati con `cp`

### $\verb|mkfs [-t type fs-options] device|$
Questo è un comando fontend (non gestiamo davvero la creazione di un filesystem) serve per creare un filesystem Linux su un `device`, tipicamente una partizione su disco rigido. Vediamo le opzioni:
- `-t type` → specifica il filesystem da usare tra [[#Tipi di filesystem|questi]] (default `ext2`)
- `fs-options` → opzioni specifiche del filesystem da passare al vero costruttore del filesystem (es. `ro` read-only, `rw` read-write)
- `device` → può essere sia il nome del dispositivo (es. `/dev/hda1/`, `/dev/sdb2`) ma anche un file
