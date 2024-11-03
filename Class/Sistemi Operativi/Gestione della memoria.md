---
Created: 2024-10-18
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Index
- [[#Roadmap|Roadmap]]
- [[#Requisiti per la gestione della memoria|Requisiti per la gestione della memoria]]
- [[#Rilocazione|Rilocazione]]
	- [[#Rilocazione#Indirizzi nei programmi|Indirizzi nei programmi]]
	- [[#Rilocazione#Soluzioni possibili|Soluzioni possibili]]
- [[#Protezione|Protezione]]
- [[#Condivisione|Condivisione]]
- [[#Organizzazione logica|Organizzazione logica]]
- [[#Gestione fisica|Gestione fisica]]
- [[#Partizionamento|Partizionamento]]
	- [[#Partizionamento#Tipi di partizionamento|Tipi di partizionamento]]
- [[#Partizionamento fisso uniforme|Partizionamento fisso uniforme]]
	- [[#Partizionamento fisso uniforme#Problemi|Problemi]]
- [[#Partizionamento fisso variabile|Partizionamento fisso variabile]]
	- [[#Partizionamento fisso variabile#Algoritmo di posizionamento|Algoritmo di posizionamento]]
	- [[#Partizionamento fisso variabile#Problemi irrisolti|Problemi irrisolti]]
- [[#Posizionamento dinamico|Posizionamento dinamico]]
	- [[#Posizionamento dinamico#Problemi|Problemi]]
- [[#Buddy System (sistema compagno)|Buddy System (sistema compagno)]]
- [[#Paginazione (semplice)|Paginazione (semplice)]]
- [[#Segmentazione (semplice)|Segmentazione (semplice)]]
	- [[#Segmentazione (semplice)#Indirizzi Logici|Indirizzi Logici]]
- [[#Concetti fondamentali|Concetti fondamentali]]
- [[#L’idea geniale|L’idea geniale]]
- [[#Esecuzione di un processo|Esecuzione di un processo]]
- [[#Conseguenze|Conseguenze]]
- [[#Memoria virtuale: terminologia|Memoria virtuale: terminologia]]
- [[#Trashing|Trashing]]
	- [[#Trashing#Principio di località|Principio di località]]
- [[#Supporto richiesto|Supporto richiesto]]
- [[#Paginazione|Paginazione]]
- [[#Traduzione degli indirizzi|Traduzione degli indirizzi]]
- [[#Overhead|Overhead]]
- [[#Tabella delle pagine a 2 livelli|Tabella delle pagine a 2 livelli]]
	- [[#Tabella delle pagine a 2 livelli#Traduzione|Traduzione]]
	- [[#Tabella delle pagine a 2 livelli#Perché conviene?|Perché conviene?]]
- [[#Translation Lookaside Buffer|Translation Lookaside Buffer]]
	- [[#Translation Lookaside Buffer#Come funziona?|Come funziona?]]
	- [[#Translation Lookaside Buffer#Memoria virtuale e process switch|Memoria virtuale e process switch]]
	- [[#Translation Lookaside Buffer#Mapping associativo|Mapping associativo]]
	- [[#Translation Lookaside Buffer#TLB e cache|TLB e cache]]
- [[#Dimensione delle pagine|Dimensione delle pagine]]
	- [[#Dimensione delle pagine#Perché grande|Perché grande]]
	- [[#Dimensione delle pagine#Perché piccola|Perché piccola]]
	- [[#Dimensione delle pagine#Dimensione delle pagine in alcuni sistemi|Dimensione delle pagine in alcuni sistemi]]
- [[#Segmentazione|Segmentazione]]
	- [[#Segmentazione#Organizzazione|Organizzazione]]
	- [[#Segmentazione#Traduzione degli indirizzi|Traduzione degli indirizzi]]
	- [[#Segmentazione#Paginazione e segmentazione|Paginazione e segmentazione]]
		- [[#Paginazione e segmentazione#Traduzione degli indirizzi|Traduzione degli indirizzi]]
	- [[#Segmentazione#Protezione e condivisione|Protezione e condivisione]]
- [[#Gestione della memoria|Gestione della memoria]]
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
Un altro problema sta nel fatto che la memoria viene utilizzata in modo inefficiente, infatti anche se un programma occupava meno memoria della dimensione della partizione, comunque gli veniva affidata una partizione intera (**frammentazione interna**)

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

### Problemi
Qui a differenza dei precedenti, si ha un problema di **frammentazione esterna**, infatti i processi, quando terminano o vengono swappati, lasciano all’interno della memoria dei “frammenti” di memoria libera. Questo è risolvibile con la **compattazione**, con la quale il SO sposta tutti i processi in modo tale che siano contigui (però ha un alto overhead)

Il SO inoltre deve scegliere a quale blocco libero assegnare un processo. Ciò si può banalmente pensare possa avvenire, come per gli altri partizionamenti, tramite l’algoritmo *best-fit*: sceglie il blocco la cui misura è la più vicina (in eccesso) a quella del processo da posizionare. Ma questo algoritmo risulta essere quello con i risultati peggiori in quanto lascia frammenti molto piccoli che costringono a fare spesso la compattazione
Come alternativa a questo algoritmo vennero proposte due alternative:
- Algoritmo *first-fit* → scorre la memoria dall’inizio, il primo blocco con abbastanza memoria viene subito scelto e per questo motivo è molto veloce. Ha come problema il fatto che tende a riempire solo la prima parte della memoria, seppur è il **migliore** tra quelli proposti
- Agloritmo *next-fit* → come il first-fit, ma anziché partire da capo ogni volta, parte dall’ultima posizione assegnata ad un processo. Sperimentalmente si nota che assegna più spesso il blocco alla fine della memoria che tendenzialmente è il più grande

>[!example] Algoritmi di posizionamento
>Memoria prima e dopo l’allocazione di un blocco da 16M
>![[Pasted image 20241021223419.png|350]]

---
## Buddy System (sistema compagno)
E’ un compromesso tra il partizionamento fisso e il partizionamento dinamico: è un partizionamento dinamico nel senso he le partizioni si creano man mano che arrivano processi, ma fisso perché non possono essere create tutte le partizioni possibili ma occorre seguire uno schema ben definito

Sia $2^U$ la dimensione dello user space e $s$ la dimensione di un processo da mettere in RAM. Quello che fa il buddy system è cominciare a dimezzare lo spazio fino a trovare un $X \text{ t.c. }2^{X-1}<s\leq 2^X \text{ con } L\leq X\leq U$ e una delle due porzioni è usata per il processo ($L$ serve per dare un lower bound per evitare che si creino partizioni troppo piccole)
Ovviamente, occorre tener presente le partizioni già occupate.
Quando un processo finisce, se il buddy è libero si può fare una fusione; la fusione può essere effettuata solo nel caso in cui è possibile costruire la partizione più grande ovvero $2^{X+1}$

>[!example]
>![[Pasted image 20241021231832.png|550]]
>
>Rappresentazione ad albero (quinta riga)
>![[Pasted image 20241021232238.png|550]]

---
# Paginazione e segmentazione
## Paginazione (semplice)
La paginazione semplice in quanto tale non è stata sostanzialmente mai usata, ma è importante a livello concettuale per introdurre la memoria virtuale.
Con la paginazione sia la memoria che i processi vengono “spacchettati” in pezzetti di dimensione uguale. Ogni pezzetto del processo è chiamato **pagina**, mentre i pezzetti di memoria sono chiamati **frame**.
Ogni pagina, per essere usata, deve essere collocata in un frame ma pagine contigue di un processo possono essere messe in un qualunque frame (anche distanti)

I SO che la adottano però devono mantenere una tabella delle pagine per ogni processo che associa ogni pagina del processo al corrispettivo frame in cui si trova.

>[!info] Quando c’è un process switch, la tabella delle pagine del nuovo processo deve essere ricaricata ed aggiornata

A differenza di prima in cui l’hardware doveva solamente intervenire e aggiungere un offset, qui deve intervenire sulle pagine stesse, infatti un indirizzo di memoria può essere visto come un numero di pagina e uno spiazzamento al suo interno (indirizzo logico)

>[!example]
>![[Pasted image 20241021233338.png|250]]
>
>Deve essere caricato un processo A che occupa 4 frame
>![[Pasted image 20241021233436.png|250]]
>
>Ne arrivano altri due da 3 (B) e 4 (C) frame 
>![[Pasted image 20241021233523.png|250]]
>
>Quindi viene swappato B
>![[Pasted image 20241021233610.png|250]]
>
>E sostituito con D (con il partizionamento dinamico, non sarebbe stato possibile caricare D in memoria)
>![[Pasted image 20241021233636.png|250]]
>
>Tabelle delle pagine risultanti
>![[Pasted image 20241021233804.png|450]]

>[!example] Esempio di traduzione
>Supponiamo che la dimensione di una pagina sia di $100 \text{ bytes}$. Quindi la RAM dell’esempio precedente è di solo $1400 \text{ bytes}$
>Inoltre i processi A, B, C, D richiedono solo $400$, $300$, $400$ e $500 \text{ bytes}$ rispettivamente (comprensivi di codice - program, dati globali ed heap - data e stack delle chiamate).
>Nelle istruzioni dei processi, i riferimenti alla RAM sono relativi all’inizio del processo (quindi, ad esempio, per D ci saranno riferimenti compresi nell’intervallo $[0, 499]$)
>
>Supponiamo ora che attualmente il processo D sia in esecuzione, e che occorra eseguire l’istruzione `j 343` (vale lo stesso anche per istruzioni di load o store, anche se devono passare per registri)
>Ovviamente non si tratta dell’indirizzo $343$ della RAM: lì c’è il processo A.
>Bisogna capire in quale pagina di D si trova $343$: basta fare `343 div 100 = 3`
>
>Poi occorre guardare la tabella delle pagine di D: la pagina $3$ corrisponde al frame di RAM numero $11$
>Il frame $11$ ha indirizzi che vanno da $1100$ a $1199$: qual è il numero giusto? Basta fare `343 mod 100 = 43` quindi il 44-esimo byte
>L’indirizzo vero è pertanto $11\cdot 100+43=1143$

>[!info]
>Per ogni processo, il numero di pagine è al più il numero di frames (non sarà più vero con la memria virtuale)
>![[Pasted image 20241024195255.png|440]]
>Per ottenere l’indirizzo vero dunque punto alla pagina formata dai $6 \text{ bit}$ più significativi, controllando la corrispondenza con il frame, e utilizzo i restanti $10 \text{ bit}$ (ogni pagina è grande $2^{10} \text{ bit}$) come offset all’interno della pagina

---
## Segmentazione (semplice)
La differenza tra paginazione e segmentazione sta nel fatto che nella paginazione le pagine sono tutte di ugual dimensione, mentre i segmenti hanno **lunghezza variabile**.
In questo risulta simile al partizionamento dinamico ma è il programmatore a decidere come deve essere segmentato il processo (tipicamente viene fatto un segmento per il codice sorgente, uno per i dati condivisi e uno per lo stack delle chiamate)

### Indirizzi Logici
![[Pasted image 20241024200117.png]]

>[!info]
>Qui si suppone che non possano esserci segmenti più grandi di $2^{12} \text{ bytes}$
>![[Pasted image 20241024201839.png|440]]
>In questo caso nella tabella delle corrispondenze oltre all’indirizzo base del segmento, ci sta anche la sua lunghezza

---
# Memoria virtuale: hardware e strutture di controllo

## Concetti fondamentali
I riferimenti alla memoria sono degli indirizzi logici che sono tradotti in indirizzi fisici a tempo di esecuzione; questo perché un processo potrebbe essere spostato più volte della memoria principale alla secondaria e viceversa durante la sua esecuzione, ogni volta occupando zone di memoria diverse.

---
## L’idea geniale
Ci si è accorti che non ci sta nessuna necessità che tutte le pagine o segmenti di un processo siano in memoria principale. Infatti per eseguire un processo ho la necessità che ci sia in memoria la pagina che contiene l’istruzione da eseguire e eventualmente i dati di cui l’istruzione ha bisogno e il resto può essere caricato in un momento successivo. Ciò fa passare dalla paginazione semplice alla paginazione con memoria virtuale.

---
## Esecuzione di un processo
Il SO porta in memoria principale solo alcuni pezzi (pagine) del programma; la porzione del processo in RAM viene chiamato *resident set*.
Se il processo tanta di accedere ad un indirizzo che non si trova in memoria viene generato un interrupt di tipo *page fault* che risulta essere una richiesta I/O a tutti gli effetti, infatti il SO mette il processo in blocked
Quindi il pezzo di processo che contiene l’indirizzo logico viene portato in memoria principale (il SO effettua una richiesta di lettura su disco) e finché quest’operazione non viene completata, un altro processo va in esecuzione. Una volta completata, un interrupt farà si che il processo torni ready (non necessariamente in esecuzione)
Quando verrà eseguito, occorrerà eseguire nuovamente la stessa istruzione che aveva causato il fault

---
## Conseguenze
Abbiamo due principali conseguenze a ciò:
- Svariati processi possono essere in memoria principale. Questo vuol dire che è molto probabile che ci sia sempre almeno un processo ready, aumentando così il grado di multiprogrammazione
- Un processo potrebbe richiedere più dell’intera memoria princiapale

---
## Memoria virtuale: terminologia
**Memoria virtuale** → schema di allocazione di memoria, in cui la memoria secondaria può essere usata come se fosse principale
- gli indirizzi usati nei programmi e quelli usati dal sistema sono diversi
- c’è una fase di traduzione automatica dai primi nei secondi
- la dimensione della memoria virtuale è limitata dallo schema di indirizzamento, oltre che ovviamente dalla dimensione della memoria secondaria
- la dimensione della memoria principale, invece, non influisce sulla dimensione della memoria virtuale
**Memoria reale** → quella principale (la RAM)
**Indirizzo virtuale** → l’indirizzo associato ad una locazione della memoria virtuale (fa sì che si possa accedere a tale locazione come se fosse parte della memoria principale)
**Spazio degli indirizzi virtuali** → la quantità di memoria virtuale assegnata ad un processo
**Spazio degli indirizzi** → la quantità di memoria assegnata ad un processo
**Indirizzo reale** → indirizzo di una locazione di memoria


>[!example] Come un processo Linux vede la memoria
>![[Pasted image 20241024234957.png|440]]

---
## Trashing
E’ ciò che succede quando il SO passa la maggior parte del tempo a swappare i processi piuttosto che ad eseguirli. Questo avviene quando un processo, la maggior parte delle richieste che fa, danno vita ad un *page fault*.
Per evitarlo, il SO cerca di indovinare quali pezzi di processo saranno usati con minore o maggiore probabilità sulla base della storia recente. Questo meccanismo sfrutta il **principio di località**

### Principio di località
I riferimenti che fa un processo fa tendono ad essere vicini (sia che si tratti di dati che di istruzioni) quindi solo pochi pezzi di processo saranno necessari di volta in volta.
Si può dunque prevedere abbastanza bene quali pezzi di processo saranno necessari nel prossimo futuro

---
## Supporto richiesto
Anche qui sarebbe necessario un eccessivo overhead per poter fare tutto, risulta dunque utile del supporto hardware.

---
## Paginazione
Ogni processo ha una sua tabella della pagine e il process control block di un processo punta a tale tabella
Le pagine contengono al loro interno, oltre che il numero di frame, anche dei bit di controllo; di questi, due sono particolarmente importanti: il **bit di presenza** (ci dice se un bit si trova in RAM oppure se è swappato) e **modified** (è zero se ci si è acceduto solo in lettura e diventa uno se è stato modificato)
![[Pasted image 20241027214737.png]]

---
## Traduzione degli indirizzi
![[Pasted image 20241027215021.png]]
La traduzione dunque è fatta dall’hardware e la somma che si vede, non è una semplice somma infatti bisogna anche moltiplicare il numero di pagina per la dimensione della pagina oltre a sommare l’indirizzo base dell’inizio del page table

Affinché questo schema funzioni, non appena un processo viene caricato la prima volta oppure si ha un process switch, il sistema operativo deve:
1. Caricare a partire da un certo indirizzo $I$ la tabella delle pagine del processo (si trova all’interno del process control block)
2. Caricare il valore di $I$ in un opportuno registro dipendente dall’hardware (`CR3` nei Pentium)

---
## Overhead
Come abbiamo detto un problema da tenere sotto controllo è quello dell’overhead, infatti le pagine potrebbero contenere molti elementi. Quando un processo è in esecuzione, viene assicurato che almeno una parte della sua tabella delle pagine sia in memoria.
Facciamo qualche conto: supponiamo $8\text{GB}$ di spazio virtuale e $1\text{kB}$ per ogni pagina, il numero di entries che ci possono essere per ogni tabella delle pagine è $\frac{2^{33}}{2^{10}}=2^{23}$ (ovvero per ogni processo)
Quanto occupa una entry? $1 \text{ byte}$ di controllo $\log_{2}(\text{size RAM in frames})$ → con massimo $4 \text{ GB}$ di RAM (architettura a 32-bits) fanno $4 \text{ bytes}$.
Max $32 \text{ bit}-10\text{ bit}=22\text{ bit}$ per i frame quindi $3 \text{ bytes}$ per il frame number, più il $\text{byte}$ di controllo
Fanno $4\cdot 2^{23}=2^{23+2}=32\text{MB}$ di overhead per ogni processo (con RAM di $1\text{GB}$, bastano $20$ processi per occupare più di metà RAM con sole strutture di overhead)

---
## Tabella delle pagine a 2 livelli
Per risolvere il problema dell’**overhead** si è pensato di fare delle tabelle delle pagine a più livelli. Anche in questo caso il processore deve avere hardware dedicato per i 2 livelli di traduzione

![[Pasted image 20241028225031.png|500]]
Il primo livello punta al secondo livello che ha sua volta punta alla memoria principale

### Traduzione
![[Pasted image 20241028225210.png]]

### Perché conviene?
Supponiamo nuovamente che abbiamo $8\text{GB}$ di spazio virtuale, vuol dire che abbiamo $33 \text{ bits}$ di indirizzo; facciamo ad es. $15\text{ bit}$ primo livello (*directory*), $8 \text{ bit}$ di secondo livello, e i rimanenti $10$ per l’offset.

>[!info]
>Spesso i processori (eg. Pentium) impongono che una tabella delle pagine di secondo livello entri in una pagina a sua volta, quello che occupa una pagina di secondo livello sta esattamente dentro una pagina di primo, infatti essa occupa $2^8\cdot 2^2=2^{10} \text{ bytes}$

Per ogni processo l’overhead è $2^{23+2}=32\text{MB}$, più l’occupazione del primo livello $2^{15+2}=128\text{kB}$.
Seppur sia aumentato lo spazio, è più facile paginare la tabella delle pagine, così in RAM basta che ci sia il primo livello più una tabella del secondo così l’overhead scende a $2^{15+2}+2^{8+2}=128\text{kB}$.
Con RAM di $1\text{GB}$, occorrono $1000$ processi per occupare più di metà della RAM con sole strutture di overhead

---
## Translation Lookaside Buffer
Translation Lookaside Buffer (TLB) letteralmente: memoria temporanea per la traduzione futura. Abbiamo visto che ogni volta che facciamo un riferimento alla memoria virtuale possono essere generati fino a due accessi alla memoria: uno per la tabella delle pagine e uno per prendere il dato

L’idea è quindi quella di usare una specie di cache per gli elementi delle tabelle delle pagine (contiene gli elementi che sono stati usati più di recente) ed è proprio il TLB

### Come funziona?
Dato un indirizzo virtuale, il processore cerca la pagina all’interno del TLB.
Se la pagina è presente (*TLB hit*), si prende il frame number e si ricava l’indirizzo reale
Se la pagina non è presente (*TLB miss*), si prende la “normale” tabella delle pagine del processo
Se la pagina risulta in memoria principale a posto, altrimenti si gestisce il page fault.
Quindi il TLB viene aggiornato includendo la pagina appena acceduta (usando un qualche algoritmo di rimpiazzamento se il TLB è già pieno: solitamente LRU)
![[Pasted image 20241028235551.png|550]]
![[Pasted image 20241028235730.png|350]]

### Memoria virtuale e process switch
Seppur il TLB sia totalmente hardware ci sono casi in cui è necessario un intervento del sistema operativo. In particolar modo il TLB deve essere resettato è infatti relativo ad un singolo processo, ma risulta essere la soluzione peggiore dal punto di vista prestazionale.

Per fare un po’ meglio, alcuni processori permettono:
- di etichettare con il PID ciascuna entry del TLB (es. Pentium), non serve fare il reset ma basta fare un confronto tra il PID attuale e quello presente nella entry del TLB
- di invalidare solo alcune parti del TLB (alla fine inefficiente)

E’ comunque necessario, anche senza TLB, dire al processore dove è la nuova tabella delle pagine (nel caso sia a 2 livelli, basta la page directory e gli indirizzi sono caricati in opportuni registri)

### Mapping associativo
La tabella delle pagine contiene tutte le pagine di un processo, mentre per quanto riguarda il TLB, essendo una cache, non può contenere un’intera tabella delle pagine e non si può usare un indice per accedervi quindi teoricamente bisognerebbe scorrerla tutta

Si può risolvere questo problema sfruttando il parallelo e controllando contemporaneamente tutte le entry del TLB. Questo supporto hardware che mi permette di fare questa ricerca veloce è chiamato **mapping associativo**

Ci sta però un altro problema, bisogna fare in modo che nel TLB contenga solo pagine in RAM, altrimenti si incorrerebbe in un page fault dopo un TLB git, ma sarebbe impossibile accorgersene (il bit di presenza potrebbe infatti essere obsoleto).
Quindi quando viene messo il bit di presenza a zero deve essere opportunamente modificato il TLB attraverso un reset parziale

![[Pasted image 20241029001206.png]]

### TLB e cache
![[Pasted image 20241029001557.png]]

---
## Dimensione delle pagine
Ma quanto dovrebbe essere la giusta dimensione di una pagina
### Perché grande
Più piccola è una pagina, minore è la frammentazione all’interno delle pagine ma è anche maggiore il numero di pagine per processo. Il che significa che è più grande la tabella delle pagine per ogni processo e quindi la maggior parte delle tabelle finisce in memoria secondaria.
La memoria secondaria è ottimizzata per trasferire grossi blocchi di dati, quindi avere le pagine ragionevolmente grandi non sarebbe male

### Perché piccola
Più piccola è una pagina, maggiore il numero di pagine che si trovano in memoria principale. Lo stesso processo può riuscire ad avere un resident set più grande facendo diminuire i page fault e aumentanto la multiprogrammazione

Per risolvere questo problema sono stati fatti diversi esperimenti per capire quale fosse il giusto compromesso
![[Pasted image 20241029002627.png]]

### Dimensione delle pagine in alcuni sistemi
![[Pasted image 20241029002941.png|center|350]]
Nelle moderne architetture hardware si possono supportare diverse dimensioni delle pagine (anche fino ad $1\text{GB}$) e il sistema operativo ne sceglie una: Linux sugli x86 va con $4\text{kB}$
Le dimensioni più grandi sono usate in sistemi operativi di architetture grandi: cluster, grandi server, ma anche per i sistemi operativi stessi (kernel mode)

---
## Segmentazione
Permette al programmatore di vedere la memoria come un insieme di spazi (segmenti) di indirizzi la cui dimensione può essere dinamica. Questo viene usato per semplificare la gestione delle strutture dati che crescono.
Permette inoltre di:
- modificare e ricompilare i programmi in modo indipendente
- condividere dati
- proteggere dati

### Organizzazione
Anche qui ogni processo ha una sua tabella dei segmenti e il process control block di un processo punta a tale tabella
Ogni entry di questa tabella contiene:
- indirizzo di partenza (in memoria principale) del segmento
- la lunghezza del segmento
- un bit per indicare se il segmento è in memoria principale o no
- un altro bit per indicare se il segmento è stato modificato in seguito all’ultima volta che è stato caricato in memoria principale
![[Pasted image 20241029003949.png]]

### Traduzione degli indirizzi
![[Pasted image 20241029004030.png]]

### Paginazione e segmentazione
La paginazione è trasparente al programmatore; il programmatore infatti non ne è a conoscenza

La segmentazione è invece visibile al programmatore (se il programma è in assembler, altrimenti ci pensa il compilatore ad usare i segmenti)
Quindi ogni segmento viene diviso in più pagine
![[Pasted image 20241029004351.png]]

#### Traduzione degli indirizzi
![[Pasted image 20241029004417.png]]

### Protezione e condivisione
Con la segmentazione, implementare protezione e condivisione. Dato che ogni segmento ha una base e una lunghezza è facile controllare che i riferimenti siano contenuti nel giusto intervallo. Per la condivisione, basta dire che uno stesso argomento serve a più processi

![[Pasted image 20241103192200.png]]

---
# Memoria virtuale e sistema operativo
## Gestione della memoria: decisioni da prendere
- Usare o no la memoria virtuale?
- Usare solo la paginazione?
- Usare solo la segmentazione?
- Usare paginazione e segmentazione?
- Che algoritmi usare per gestire i vari aspetti della gestione della memoria?

---
## Elementi centrali per il progetto del SO
Quando si progetta un sistema operativo è necessario decidere le seguenti cose:
- Politica di prelievo (*fetch policy*)
- Politica di posizionamento (*placement policy*)
- Politica di sostituzione (*replacement policy*)
- Altro (gestione del resident set, politica di pulitura, controllo del carico)
Il tutto, cercando di minimizzare i page fault; non c’è una politica sempre vincente

## Fetch policy
Decide quando una pagina data deve esser portata in memoria principale.

Si usano principalmente due politiche:
- paginazione su richiesta (*demand paging*)
- prepaginazione (*prepaging*)

### Demand paging
Una pagina viene portata in memoria principale nel momento in cui qualche processo la richiede. Ciò causa molti page fault nei primi momenti di vita del processo

### Prepaging
Cerca di anticipare le necessità del processo. Questa politica infatti porta in memoria principale più pagine di quelle richieste (ovviamente si tratta di pagine vicine a quella richiesta)

---
## Placement policy
La placement policy decide in quale frame mettere una pagina una volta che è stata prelevata dal disco. Seppur tramite la traduzione degli indirizzi la si può posizionare ovunque, tipicamente una pagina viene posizionata nel **primo** (con indice numericamente più basso) **frame libero**.

>[!hint]
>Questa politica si applica quando ci sta almeno un frame libero in RAM, se non ne è disponibile nessuno si parlerà di *replacement policy*

---
## Replacement policy
Questa viene applicata quando è stato prelevato una pagina dal disco, ma non è disponibile alcun frame in RAM in cui posizionarla
Essenzialmente, una volta deciso quale è il frame giusto da sostituire tramite un algoritmo di replacement policy (generalmente si cerca di minimizzare la possibilità che la pagina appena sostituita venga richiesta di nuovo, usando il principio di località, si cerca di predire il futuro sulla base del passato recente) e inoltre è necessario aggiornare la tabella della pagine. Nella pagina prelevata dal disco viene impostato il bit di presenza a uno mentre per la pagina da sostituire il bit di presenza viene impostato a zero

### Frame bloccati
Bisogna tenere presente, nella replacement policy, che alcuni frame potrebbero essere bloccati, attraverso un bit hardware gestito dal SO. Tipicamente questo stato riguarda frame del sistema operativo stesso, oppure di processi che potrebbero riguardare trasferimenti di I/O

---
## Gestione del resident set
La gestione del resident set risponde a 2 necessità:
- decidere per ogni processo che è in esecuzione quanti frame vanno allocati (*resident set management*)
- decidere se, quando si rimpiazza un frame, è possibile sostituire solo un frame che fa parte dello stesso processo oppure se si può sostituire un frame qualsiasi (*replacement scope*)

### Resident set management
Sono presenti due possibilità per decidere quanto spazio dedicare ad ogni singolo processo in RAM:
- **allocazione fissa** → il numero di frame è deciso a tempo di creazione del processo
- **allocazione dinamica** → il numero di frame è deciso durante la vita del processo (magari basandosi su statistiche che man mano vengono raccolte)

![[Pasted image 20241103195411.png]]
Ovviamente, con resident set alto si ha ottimi page fault rate, ma poca multiprogrammazione

### Replacement scope
Anche qui si hanno due possibilità:
- **politica locale** → se bisogna rimpiazzare un frame, si sceglie un altro frame dello stesso processo
- **politica globale** → si può scegliere qualsiasi frame (ovviamente non del SO

In tutto si hanno 3 possibili strategie, infatti con l’allocazione fissa, la politica globale non si può usare altrimenti si potrebbe ampliare il numero di frame do un processo, e non sarebbe più allocazione fissa

---
## Politica di pulitura
Se un frame è stato modificato, va riportata la modifica anche sulla pagina corrispondente.
Anche qui si hanno due possibilità per decidere quando riportare questa modifica:
- non appena avviene la modifica
- non appena il frame viene sostituito

Tipicamente si fa una via di mezzo, intrecciata con il *page buffering* (concetto di I/O che vedremo); solitamente, si raccolgono un po’ di richieste di frame da modificare e le si esegue

---
## Controllo del carico (medium term scheduler)
![[Pasted image 20241104001407.png]]
Lo scopo del *medium term scheduler* è quello di **controllo del carico**, ovvero mantenere il livello di multiprogrammazione il più alto possibile ma senza arrivare al trashing (ottimizzando il page fault rate)

Per farlo il medium term scheduler ha due possibilità: o sospendere un processo, oppure metterlo in RAM. 
### Stati dei processi e scheduling
![[Pasted image 20241104002005.png]]
Adesso possiamo specificare meglio cosa vuol dire che un processo è suspended, vuol dire che il suo resident set è zero (non ci sono pagine in RAM). Mentre ready vuol dire che una parte del processo è in RAM (almeno una pagina).
Uno dei motivi per cui un processo diventa suspended è a causa del medium term scheduler.