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
Il programmatore (assembler o compilatore) non sa e non deve sapere in quale zona della memoria il programma verrà caricato. Questo atteggiamento è chiamato **rilocazione**, il programma deve essere in grado di **essere eseguito indipendentemente da dove si trovi in memoria**.
Può accadere infatti che:
- potrebbe essere swappato su disco, e al ritorno in memoria principale potrebbe essere in un’altra posizione
- potrebbe anche non essere contiguo, oppure con alcune pagine in RAM e altre su disco

I riferimenti alla memoria devono tradotti nell’indirizzo fisico “vero”; può essere fatto tramite preprocessing o **runtime** (ogni volta che viene eseguita un’istruzione, se quell’istruzione contiene un indirizzo occorre fare la sostituzione), in quest’ultimo caso è necessario avere un supporto hardware (a livello software ci sarebbe un overhead troppo grande)

![[Pasted image 20241018220727.png|400]]

### Indirizzi nei programmi
Il *linker* ha il compito di unire quelli che sono i moduli (ovvero tutti i file di un programma) e le librerie statiche (librerie esterne) e crea come output un *load module* che può essere preso e caricato in memoria RAM, passaggio che avviene tramite il *loader* che unisce eventuali librerie dinamiche
![[Pasted image 20241018221354.png|450]]

Si parte dal caso (a) in cui si hanno dei link simbolici alle parti del programma
Nel caso (b) si hanno degli indirizzi assoluti ma questo è possibile solo nel caso in cui si sa da che indirizzo si parte, deve quindi avvenire in preprocessing
Nel caso (c) si salta ad un indirizzo relativo rispetto all’inizio del programma
Nel caso (d) si salta ad un indirizzo relativo rispetto alla posizione del programma in memoria
![[Pasted image 20241018221459.png|650]]

Abbiamo quindi tre tipi di indirizzi:
- **Logici** → il riferimento in memoria è indipendente dall’attuale posizionamento del programma in memoria
- **Relativi** → riferimento è espresso come uno spiazzamento (differenza) rispetto ad un qualche punto noto (caso particolare degli indirizzi logici)
- **Fisici o Assoluti** → riferimento effettivo alla memoria

### Soluzioni possibili
Vecchissima soluzione: gli indirizzi assoluti vengono determinati nel momento in cui il programma viene caricato (nuovamente o per la prima volta) in memoria
Soluzione più recente: gli indirizzi assoluti vengono determinati nel momento in cui si fa un riferimento alla memoria (serve hardware dedicato).
E’ necessario un hardware dedicato in quanto, se non ci fosse, ogni volta che un processo viene riportato in memoria, potrebbe essere in un posto diverso. Nel frattempo, potrebbero essere arrivati altri processi e averne preso il posto, quindi, ad ogni ricaricamento in RAM, occorre ispezionare tutto il codice sorgente del processo sostituendo man mano tutti i riferimenti agli indirizzi. Ciò risulterebbe in troppo overhead

![[Pasted image 20241018222723.png|350]]
Base register (registro base) → indirizzo di partenza del processo
Bounds register (registro limite) → indirizzo di fine del processo
I valori per questi registri vengono settati nel momento in cui il processo viene posizionato in memoria mantenuti nel PCB del processo fa parte del passo 6 per il process switch (vedere slides sui processi); non vanno semplicemente ripristinati: occorre proprio modificarli

