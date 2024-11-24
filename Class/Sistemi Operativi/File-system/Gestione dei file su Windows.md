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

### Struttura
![[Pasted image 20241124221953.png|center]]

#### Boot sector
Questa regione contiene le informazioni necessarie per l’accesso al volume ovvero il tipo e puntatore alle altre sezioni del volume, e il bootloader del sistema operativo in BIOS/MBR
#### FAT
Questa regione contiene due copie della file allocation table (utile in caso la tabella principale si corrompa) che sono sincronizzate ad ogni scrittura su disco
Permette, come già detto, di mappare il contenuto della regione dati, indicando a quali directory/file i diversi cluster appartengono

### Funzionamento
