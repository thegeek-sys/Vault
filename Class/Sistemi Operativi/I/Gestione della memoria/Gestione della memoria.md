## Perché gestire la memoria (nel SO)?
La memoria è oggi a basso costo e con un trend in diminuzione, questo in quanto i programmi occupano sempre più memoria.
Se il SO lasciasse a ciascun processo la gestione della propria memoria, ogni processo la prenderebbe per intera, che chiaramente va contro la multiprogrammazione, se ci sono $n$ processi ready, ma uno solo di essi occupa l’intera memoria risulterebbe inutile.
Si potrebbe quindi imporre, come avveniva negli anni ‘70, dei limiti di memoria a ciascun processo, però risulterebbe difficile per un programmatore rispettare tali limiti

La soluzione è quindi farla gestire al sistema operativo cercando però di dare l’illusione ai processi di avere tutta la memoria a loro disposizione. Per farlo viene utilizzato il disco come buffer per i processi (se solo 2 processi entrano in memoria, gli altri 8 saranno swappati su disco). Dunque il SO deve pianificare lo swap in modo intelligente, così da massimizzare l’efficienza del processore

In conclusione occorre gestire la memoria affinché ci siano sempre un numero ragionevole di processi pronti all’esecuzione, così da non lasciare inoperoso il processore.

- [[Gestione della memoria - requisiti di base]]
- [[Partizionamento della memoria]]
- [[Paginazione e segmentazione]]
- [[Memoria virtuale - hardware e strutture di controllo]]
- [[Memoria virtuale e sistema operativo]]
- [[Gestione della memoria in Linux]]