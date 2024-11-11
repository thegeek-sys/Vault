---
Created: 2024-11-12
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Page cache in Linux
Analizziamo adesso la page cache in Linux ([[Cache del disco#Introduction|qui]]).
In Linux la page cache è unica per tutti i trasferimenti tra disco e memoria, compresi quelli dovuti alla gestione della memoria virtuale (una sorte di page buffering)
Questa permette due vantaggi:
- condensare le scritture
- sfruttare la località dei riferimenti (si risparmiano accessi a disco)

Con questa si scrive su disco quando:
- è rimasta poca memoria → una parte della page cache è ridestinata ad un uso diretto dei processi (si può allargare e restringere in base alle richieste)
- quando l’età delle pagine “sporche” va sopra una certa soglia

Non è inoltre presente una politica separata di replacement: è infatti la stessa usata per il rimpiazzo delle pagine (la page cache è paginata, e le sue pagine sono rimpiazzate con l’algoritmo visto per la gestione della memoria)