---
Created: 2024-03-11
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

## Introduction
La codifica dell’istruzione deve indicare
- quale operazione va svolta (opcode)
- quali argomenti sono necessari
- dove scrivere il risultato

---
## Modi di indirizzamento
- **implicito** (0 accesso alla memoria) → sorgente/destinazione fissa
- **immediato** (0 accessi alla memoria) → non chiedo al processore di leggere qualcosa dai registri ma di leggere il valore che gli viene passato nell’istruzione
- **diretto** (1 accesso alla memoria) → dentro l’istruzione ci sta scritto dove leggere le informazioni dalla memoria (ci sono dei bit che indicano l’indirizzo in memoria)
- **indiretto** (2 accessi alla memoria) → serve un valore il cui indirizzo è scritto in un altro indirizzo
- **a registro** (0 accessi alla memoria) → leggo direttamente un registro
- **a registro indiretto** (1 accesso alla memoria) → accesso al registro e accesso alla memoria (in un registro ci sta scritto un indirizzo di memoria)