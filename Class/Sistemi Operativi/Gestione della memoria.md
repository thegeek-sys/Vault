---
Created: 2024-10-18
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Perché gestire la memoria (nel SO)?
La memoria è oggi a basso costo e con un trend in diminuzione, questo in quanto i programmi occupano sempre più memoria.
Se il SO lasciasse a ciascun processo la gestione della propria memoria, ogni processo la prenderebbe per intera, che chiaramente va contro la multiprogrammazione, se ci sono $n$ processi ready, ma uno solo di essi occupa l’intera memoria risulterebbe inutile.
Si potrebbe quindi imporre, come avveniva negli anni ‘70, dei limiti di memoria a ciascun processo, però risulterebbe difficile per un programmatore rispettare tali limiti

La soluzione è quindi farla gestire al sistema operativo cercando però di dare l’illusione ai processi di avere tutta la memoria a loro disposizione. Per farlo viene utilizzato il disco come buffer per i processi (se solo 2 processi entrano in memoria, gli altri 8 saranno swappati su disco). Dunque il SO deve pianificare lo swap in modo intelligente, così da massimizzare l’efficienza del processore

In conclusione occorre gestire la memoria affinché ci siano sempre un numero ragionevole di processi pronti all’esecuzione, così da non lasciare inoperoso il processore.

---
## Requisiti per la gestione della memoria
I requisiti per la gestione della memoria sono:
- **Rilocazione** → richiede che ci sia aiuto hardware (il sistema operativo viene aiutato da opportune funzioni macchina di basso livello)
- **Protezione** → richiede che ci sia aiuto hardware
- **Condivisione**
- **Organizzazione logica**

---
## Rilocazione
Il programma non sa e non deve sapere in quale zona della memoria il programma verrà caricato. Questo atteggiamento è chiamato **rilocazione**, il programma deve essere in grado di **essere eseguito indipendentemente da dove si trovi in memoria**.
Può accadere infatti che:
- potrebbe essere swappato su disco, e al ritorno in memoria principale potrebbe essere in un’altra posizione
