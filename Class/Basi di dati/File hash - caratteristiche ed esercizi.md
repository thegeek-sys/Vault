---
Created: 2024-12-11
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
La particolarità dei file hash è il fatto che il file è suddiviso in **bucket** numerati da $0$ a $B-1$. Ciascun bucket è costituito da **uno o più blocchi** collegati mediante puntatori ed è organizzato come un heap

---
## Bucket
![[Pasted image 20241211001548.png|center|600]]
### Bucket directory
L’accesso ai bucket avviene attraverso la **bucket directory** che contiene $B$ elementi. L’$i$-esimo elemento contiene l’indirizzo del primo blocco dell’$i$-esimo bucket (**bucket header**)
![[Pasted image 20241211001855.png|center|600]]

---
## Funzione di hash
Dato un valore $v$ per la chiave, il numero del bucket in cui deve trovarsi un record con chiave $v$ è calcolato mediante una funzione che prende il nome di **funzione di hash**

---
## Operazioni
Una qualsiasi operazione (ricerca, inserimento, cancellazione )

