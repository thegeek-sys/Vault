---
Created: 2024-11-24
Class: "[[Sistemi Operativi]]"
Related:
  - "[[File-system]]"
Completed:
---
---
## Introduction
In Windows esistono due tipi di file-system:
- FAT (vecchio, da MS-DOS) → allocazione concatenata, con blocchi (*cluster*) di dimensione fissa
- NTFS (nuovo) → allocazione con **bitmap**, con blocchi (*cluster*) di dimensione fissa

---
## Caratteristiche principali di FAT
FAT è l’acronimo di *File Allocation Table* ovvero una tabella ordinata di puntatori alla memoria.
Come file-system risulta essere molto limitato per gli usi attuali ma andava bene per i vecchi dischi (soprattutto per i floppy), seppur rimanga ancora usato sulle chiavette USB.

L’approccio però è non scalabile e dunque la FAT stessa può occupare molto spazio

---
## File Allocation Table
Ogni riga della FAT è un puntatore ad un **cluster** (blocco) del disco. Ogni cluster ha dimensione variabile (può cambiare da partizione a partizione, ma resta fisso all’interno di una partizione) tra $2$ e $32 \text{ KB}$, ed è costituito da settori di disco contigui.
Come conseguenza si ha che la tabella cresce con la grandezza della partizione.
I puntatori sono valori di $12$, $16$ o $32 \text{ bit}$ (FAT-12, FAT-16, FAT-32).

La parte della FAT relativa ai file aperti deve essere sempre mantenuta interamente in memoria principale; questa parte consente infatti di identificare i blocchi di un file e accedervi seguendo sequenzialmente i collegamenti nella FAT (un file è semplicemente una catena di indici, risulta essere un misto tra allocazione concatenata e indicizzata)

In questo modo però risulta avere overhead di spazio non indifferente infatti ci sta un puntatore ($32 \text{ bit}$) per riga, e tante righe quanti cluster del disco. Con un disco da $100\text{ GB}$ e cluster da $1\text{ KB}$, la FAT ha $100$ milioni di righe, dunque $4\text{B}\cdot 1.000.000 =400\text{ MB}$ per la FAT

### Funzionamento
Data la tabella FAT di puntatori, il puntatore i-esimo:
- se è tutti zero → l’i-esimo cluster del disco è libero
- se è tutti uno → valore speciale, vuol dire che è l’ultimo blocco del file
- se non è zero e non è un valore speciale → indica il cluster dove trovare il prossimo pezzo del file, oltre che la prossima entry della FAT per questo file

![[Pasted image 20241124223239.png|600]]
Ci sta una corrispondenza uno ad uno con i cluster nel disco. Ad esempio il FILE3 punta alla settima entry del FAT, in cui ci stanno memorizzati tutti 1; il che vuol dire che il settimo cluster è l’unico in cui ci sono i dati relativi al FILE3

### Struttura
![[Pasted image 20241124221953.png|center]]

#### Boot sector
Questa regione contiene le informazioni necessarie per l’accesso al volume ovvero il tipo e puntatore alle altre sezioni del volume, e il bootloader del sistema operativo in BIOS/MBR
#### FAT
Questa regione contiene due copie della file allocation table (utile in caso la tabella principale si corrompa) che sono sincronizzate ad ogni scrittura su disco
Permette, come già detto, di mappare il contenuto della regione dati, indicando a quali directory/file i diversi cluster appartengono
#### Regione Root Directory
E’ una *directory table* che contiene tutti i file entry per la directory root di sistema (che ha dimensione fissa e limitata in FAT12 e FAT16, $265 \text{ entries}$)
In FAT32 è inclusa nella regione dati, insieme a file e directory normali, e non ha limitazioni sulla dimensione