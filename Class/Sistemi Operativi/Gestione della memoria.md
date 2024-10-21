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
## Roadmap
- Gestione della memoria: requisiti di base
- Partizionamento della memoria
- Paginazione e segmentazione
- Memoria virtuale: hardware e strutture di controllo
- Memoria virtuale e sistema operativo
- Gestione della memoria in Linux

---
# Gestione della memoria: requisiti di base
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

Dunque il valore del registro base viene aggiunto al valore dell’indirizzo relativo per ottenere l’indirizzo assoluto e il risultato viene confrontato con il registro limite. Se va oltre, viene generato un interrupt per il SO (simile al segmentation fault)

---
## Protezione
I processi non devono poter accedere a locazioni di memoria di un altro processo, a meno che non siano autorizzati. Anche questo, a causa della rilocazione, non può essere fatto a tempo di compilazione, deve quindi esser fatto a tempo di esecuzione. Per questo motivo è necessario aiuto hardware

---
## Condivisione
Solitamente la maggior parte dell’immagine del processo deve essere protetta, ma ci possono essere dei casi (es. se può processi devono risolvere uno stesso problema) ci può essere una parte del processo accessibile sia in lettura sia in scrittura da un altro processo.

Dunque può avvenire sia perché il programmatore ne è cosciente, sia può essere fatto direttamente dal sistema operativo. Un caso tipico è quando più processi vengono creati eseguendo più volte lo stesso codice sorgente (fintantoché questi processi restano in esecuzione, è più efficiente che condividano il codice sorgente, visto che è lo stesso)

---
## Organizzazione logica
A livello hardware la memoria è organizzata in modo lineare (sia RAM che disco), mentre a livello software i programmi sono tipicamente scritti in moduli, che possono essere compilati separatamente e che possono avere permessi diversi

Questo requisito fa si che il SO faccia il **bridging** (traduzione) tra quello che ad alto livello vedono i programmatori e ciò che effettivamente viene eseguito (spesso tramite segmentazione)

---
## Gestione fisica
La gestione fisica serve a gestire il flusso di dati tra RAM (piccola, veloce e volatile) e memoria secondaria (grande, lenta permanente).

Fino a circa 40 anni fa la gestione della RAM veniva lasciata al programmatore, doveva infatti gestire esplicitamente lo swapping in base a quanta memoria gli veniva messa a disposizione dal SO. Risulta però un processo molto complesso, per questo è stato deciso di affidare questo compito esclusivamente al SO

---
# Partizionamento della memoria
## Partizionamento
Uno dei primi metodi per la gestione della memoria è il **partizionamento**, ma è comunque utile per capire la memoria virtuale (la memoria virtuale è l’evoluzione moderna delle tecniche di partizionamento)
### Tipi di partizionamento
- Partizionamento fisso
- Partizionamento dinamico
- Paginazione semplice
- Segmentazione semplice
- Paginazione con memoria virtuale
- Segmentazione con memoria virtuale

---
## Partizionamento fisso uniforme
Con il partizionamento fisso, il SO, all’avvio, segmenta la memoria in partizioni di ugual lunghezza. Dunque in ognuna partizione posso mettere un processo che occupa al più la dimensione della partizione. Il sistema operativo può decidere di swappare un processo per toglierlo da un segmento (es. suspended)
![[Pasted image 20241021130138.png|100]]
### Problemi
Quando un processo era troppo grande per la partizione, il programmatore doveva usare la tecnica dell’*overlays* (gestire esplicitamente lo swap) per fare in modo di non occupare più memoria di quella disponibile
Un altro problema sta nel fatto che la memoria viene utilizzata in modo inefficiente, infatti anche se un programma occupava meno memoria della dimensione della partizione, comunque gli veniva affidata una partizione intera

---
## Partizionamento fisso variabile
Nel partizionamento fisso variabile, così come nel partizionamento fisso, le partizioni vengono create all’inizializzazione del sistema operativo, ma a differenza di prima qui le partizioni non hanno tutte la stessa dimensione, mitigando quindi i problemi del partizionamento precedente (senza però risolverli)
![[Pasted image 20241021212807.png|100]]

### Algoritmo di posizionamento
Per capire in quale partizione posizionare un determinato processo è necessario un algoritmo di posizionamento. Un processo infatti, in questo caso, viene posizionato nella partizione più piccola che può contenerlo, minimizzando la quantità di spazio sprecato

Ciò ci pone però di fronte a due scelte: o utilizzo una coda per ogni partizione oppure utilizzo una singola coda per tutti i processi e solo alla fine decido in quale partizione posizionarlo
![[Pasted image 20241021213224.png|500]]

### Problemi irrisolti
Seppur abbia mitigato i problemi del partizionamento precedente, lascia dei problemi irrisolti.
C’è un numero massimo di processi in memoria principale (corrispondente al numero di partizioni deciso inizialmente). Se ci sono molti processi piccoli, la memoria verrà usata in modo inefficiente (seppur questo problema è risolvibile utilizzando una sola coda per i processi e mettendo il processo in una partizione più grande nel caso in cui quella in cui dovrebbe andare è già occupata)

---
## Posizionamento dinamico
In questo tipo di partizionamento, le partizioni variano sia in misura che in quantità, allocando per ciascun processo esattamente la quantità di memoria che serve

>[!example]- Esempio
>### Esempio
![[Pasted image 20241021213901.png|100]]
Quando il sistema operativo viene inizializzata la memoria senza alcun processo al suo interno
>
![[Pasted image 20241021213949.png|100]]
In un certo momento arrivano 3 processi che lasciano 4M di memoria libera
>
>![[Pasted image 20241021214050.png|100]]
Ad un certo momento arriva un processo P4 che richiede 8M, quindi il sistema operativo decide che per qualche motivo P4 è più importante di P2, che viene swappato per mettere al suo posto P4
>
>![[Pasted image 20241021214211.png|100]]
Quindi è il momento di P1 di essere swappato su disco per mettere al suo posto P2
Se per caso arriva un processo da 8M questo non potrà essere inserito in quanto la memoria a disposizione non è contigua, sarebbe quindi necessario rimuovere un altro processo per fargli spazio
