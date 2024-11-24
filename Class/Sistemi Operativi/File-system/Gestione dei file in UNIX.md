---
Created: 2024-11-24
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
In Unix ci sono sei tipi di file:
- normale
- directory
- speciale (mappano su nomi di file i dispositivi di I/O)
- named pipe (per far comunicare i processi tra loro)
- hard link (collegamenti, nome di file alternativo)
- link simbolico (il suo contenuto è il nome del file cui si riferisce)

---
## Inode
Inode sta per *”index node”* (ispirato al metodo di [[Gestione della memoria secondaria#Allocazione indicizzata#Porzioni di lunghezza fissa|allocazione indicizzato]], con dimensione fissa dei blocchi)
Questa struttura dati contiene le informazioni essenziali per un dato file, ma un dato inode potrebbe essere associato a più nomi di file

### Inode in Free BSD
 