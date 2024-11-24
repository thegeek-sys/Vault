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
