---
Created: 2024-10-14
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
Un sistema operativo deve allocare risorse tra diversi processi che ne fanno richiesta contemporaneamente. Tra le diverse possibili risorse, c’è il tempo di esecuzione, che viene fornito da un processore. Questa risorsa viene allocata tramite lo **scheduling**

### Scopo dello scheduling
Dunque lo scopo dello scheduling è quello di assegnare ad ogni processore i processi da eseguire, man mano che i processi stessi vengono creati e distrutti. Tale obiettivo va raggiunto ottimizzando vari aspetti:
- tempo di risposta
- throughput
- efficienza del processore

### Obiettivi dello scheduling
L’obiettivo è dunque quelli di distribuire il tempo di esecuzione in modo equo tra i vari processi, ma al tempo stesso anche gestire le priorità dei processi quando necessario (es. vincoli di real time ovvero eseguire operazioni entro un certo tempo).
Deve inoltre evitare la starvation dei processi, ma anche avere un **overhead** basso, ovvero avere un tempo di esecuzione dello scheduler stesso basso

---
## Tipi di Scheduling
Esistono vari tipi di scheduler in base a quanto spesso un processo viene eseguito:
- *Long-term scheduling* → decide l’aggiunta ai processi da essere eseguiti (eseguito molto di rado)
- *Medium-term scheduling* → decide l’aggiunta ai processi che sono in memoria principale (eseguito sempre non molto spesso)
- *Short-term scheduling* → decide quale processo, tra quelli pronti, va eseguito da un processore (eseguito molto spesso)
- *I/O scheduling* → decide a quale processo, tra quelli con una richiesta pendente per l’I/O, va assegnato il corrispondente dispositivo di I/O

---
## Processi e scheduling
### Stati dei processi
Ricordando il modello a 7 stati, le transizioni tra i vari stati sono decise dallo scheduler
![[Pasted image 20241014213242.png|500]]
Il long-term scheduler si occupa della creazione dei nuovi processi
Il mid-term scheduler si occupa dello swapping tra memoria principale e disco (e viceversa)
Lo short-term scheduler si occupa di decidere quali processi ready devono essere running

>[!info]- Modello a 7 stati
>![[Processi#Processi sospesi]]

### Code dei processi
![[Pasted image 20241014213625.png|550]]

---
## Long-term scheduling
Il long-term scheduling decide quali programmi sono ammessi nel sistema per essere eseguiti. Questo tipicamente è FIFO (first in first out) dunque il primo che è arriva è il primo ad essere ammesso, seppur tenga conto di criteri come priorità, requisiti per I/O ecc.
Controlla quindi il grado di multiprogrammazione, e all’aumentare del numero di processi, diminuisce la percentuale di tempo per cui ogni processo viene eseguito

Tipiche strategie
- i lavori batch (non interattivi) vengono accodati e il LTS li prende man mano che lo ritiene “giusto”
- i lavori interattivi vengono ammessi fino a “saturazione del sistema”
- se si sa quali processi sono I/O-bound e quali CPU-bound (quali usano più la cpu e quali più l’i/o) mantiene un giusto mix tra i due tipi; oppure se si sa quali processi fanno richieste a quali dispositivi di I/O, fare in modo da bilanciare tali richieste

In generale si può dire che il LTS viene chiamato in causa quando vengono creati dei processi ma ad esempio interviene anche quando termina un processo o quando alcuni processi sono idle da troppo tempo

---
## Medium-term scheduler
Il medium-term scheduler è parte integrante della funzione di swapping per i processi. Il passaggio da memoria secondaria a principale è basato sulla necessità di gestire il grado di multiprogrammazione

---
## Short-term scheduler
Lo short-term scheduler è chiamato anche *dispatcher*, è quello eseguito più frequentemente, ed è invocato sulla base di eventi che accadono:
- interruzioni di clock
- interruzioni di I/O
- chiamate di sistema
- segnali

### Scopo
Lo scopo dello short-term scheduler è quello di allocare il tempo di esecuzione su un processore per ottimizzare il comportamento dell’intero sistema, dipendentemente da terminati indici prestazionali. Per valutare una data politica di scheduling, occorre prima definire dei criteri

### Criteri
Occorre distinguere tra criteri per l’utente e criteri per il sistema prestazionali e non prestazionali.
Quelli prestazionali sono quantitativi e facili da misurare
Quelli non prestazionali sono qualitativi e difficili da misurare
#### Criteri utente
Prestazionali:
- **Turnaround time**
- **Response time**
- **Deadline**
Non prestazionali:
- **Predictability**
##### Turn-around Time
Tempo per la creazione di un processo (sottomissione) e il suo completamento, compresi tutti i vari tempi di attesa (I/O, processore). Spesso viene utilizzato per processi batch (non interattivi)
##### Response time
Qui si trattano quasi esclusivamente processi interattivi. Si tratta del tempo che intercorre tra la sottomissione di una richiesta (l’utente fa un’interazione) e l’inizio del tempo di risposta.
Per il response time lo scheduler ha un duplice obiettivo
- minimizzare il tempo di risposta medio
- massimizzare il numero di utenti con un buon tempo di risposta
##### Deadline e Predictability
Per deadline si intende, nei casi in cui un processo specifica una scadenza, lo scheduler dovrebbe come prima cosa massimizzare il numero di scadenze rispettate
Per predictability si intende che non deve esserci troppa variabilità nei tempi di risposta e/o di ritorno, a meno che il sistema non sia completamente saturo

#### Criteri sistema
Prestazionali:
- **Throughput**
- **Processor utilization**
Non prestazionali
- **Fairness**
- **Enforcing priorities**
- **Balancing resources**
##### Throughput
Ricordiamo il throughput è il numero di processi in grado di eseguire in una determinata unità di tempo, e il dispatcher ha il compito di massimizzarlo (nonostante in parte dipenda anche dal tempo richiesto dal processo)
##### Processor utilization
Il processor utilization è la percentuale di tempo in cui il processore viene utilizzato, si cerca dunque di minimizzare il tempo in cui il processore è idle
Particolarmente utile per sistemi costosi, condivisi tra più utenti
##### Bilanciamento delle risorse
Lo scheduler deve far si che le risorse del sistema siano usate il più possibile. Infatti i processi che useranno meno le risorse attualmente più usate dovranno essere favoriti
##### Fairness e priorità
Se non ci sono indicazioni dagli utenti o dal sistema (es. non c’è priorità), tutti i processi devono essere trattati allo stesso modo, dunque a tutti i processi deve essere data la stessa possibilità di andare in esecuzione (evitando la starvation).
Se invece ci sono priorità, lo scheduler deve favorire i processi a priorità più alta (occorre avere più code, una per ogni livello di priorità)
![[Pasted image 20241014222223.png|370]]
Potrebbe però nascere un problema: un processo con priorità più bassa potrebbe soffrire di starvation; la soluzione sta nell’aumentare la priorità del processo a mano a mano che l’”età” del processo aumenta

---
## Politiche di scheduling
![[Pasted image 20241014223048.png]]
Le colonne rappresentano gli algoritmi per la politica di scheduling, nei sistemi moderni vengono attuate più politiche insieme

### Funzione di selezione
La funzione di selezione è quella che sceglie effettivamente il processo da mandare in esecuzione. Se è basata sulle caratteristiche dell’esecuzione, i parametri da cui dipende sono:
- *w* → tempo trascorso in attesa
- *e* → tempo trascorso in esecuzione
- *s* → tempo totale richiesto, incluso quello già servito (quindi va stimato o fornito come input insieme alla richiesta di creazione del processo)

### Modalità di decisione
La modalità di decisione specifica in quali istanti di tempo la funzione di selezione viene invocata. Ci sono due possibilità:
- **preemptive**
	Il sistema operativo può interrompere un processo indipendentemente da eventi terzi (lo può decidere per conto suo). In questo caso il processo diverrà ready.
	Questa può avvenire o per l’arrivo di nuovi processi (appena forkati) o per un interrupt (può essere di I/O, un processo blocked diventa ready, o di clock, periodico per evitare che un processo monopolizzi il sistema)
- **non-preemptive**
	Se un processo è in esecuzione, allora arriva o fino a terminazione o fino ad una richiesta di I/O (o comunque ad una richiesta bloccante)

### Scenario comune di esempio
![[Pasted image 20241014224337.png|550]]

---
## FCFS (First Come First Served)
La politica FCFS è una politica non-preemptive. Quando un processo smette di essere eseguito, si passa al processo che ha aspettato di più nella coda ready finora
![[Pasted image 20241014224613.png|400]]
Il problema di questo tipo di politica è che un processo “corto” potrebbe dover attendere molto prima di essere eseguito (come è capitato per il processo E nell’esempio). Tende inoltre a favorire i processi che usano molto da CPU (CPU-bound), che infatti, una volta preso possesso della CPU, non viene rilasciata finché il processo non termina

---
## Round-Robin
La politica Round-Robin è una politica preemptive, basandosi sul clock. In questa politica è infatti necessario fissare un’unità di tempo che determina il tempo di esecuzione di ogni processo
![[Pasted image 20241014225032.png|400]]
Nella pratica un’interruzione di clock viene generata ad intervalli periodici, e quando questa arriva il processo attualmente in esecuzione viene rimesso nella coda dei ready (ovviamente sei processo in esecuzione arriva ad un’istruzione I/O prima dell’interruzione allora viene spostato nella coda dei blocked) e il prossimo processo ready nella coda viene selezionato
![[Pasted image 20241014231706.png|400]]

### Misura del quanto di tempo per la preemption
Il quanto di tempo deve essere poco più grande del “tipico“ tempo di interazione di un processo (tempo che ci mette un processo a rispondere, il round-robin è tipico di processi interattivi)
![[Pasted image 20241014232145.png|200]]

Se invece si scegliesse un quanto di tempo minore del tipico tempo di interazione il tempo di risposta per un processo aumenta notevolmente (vengono infatti mandati in esecuzione tutti gli altri processi prima di calcolare la risposta)
![[Pasted image 20241014232323.png]]

Ma se lo si fa troppo lungo, potrebbe durare più del tipico processo e il round-robin degenera in FCFS

### CPU-bound vs. I/O-bound
I processi CPU-bound con il round-robin sono favoriti, infatti vuol dire che il proprio quanto di tempo viene usato per intero o quasi. Invece gli I/O bound ne usano solo una porzione, infatti non appena arriva una richiesta bloccante, il processo va nella coda dei blocked.
Risulta quindi essere non equo e non efficiente per l’I/O

![[Pasted image 20241014233624.png|360]]
Come soluzione è stato proposto il round-robin **virtuale**; infatti se un processo fa una richiesta bloccante, una volta che è stata esaudita, non va nella coda dei ready come accadrebbe solitamente, bensì viene direttamente messo in una coda prioritaria che viene scelta per prima dal dispatcher e vengono eseguiti per il quanto di tempo rimanente dalla precedente esecuzione prima di essere bloccato

---
## SPN (Shortest Process Next)
La politica shortest process next è una politica non-preemprive. Questa manda in esecuzione il processo con tempo di esecuzione più breve (tempo di esecuzione stimato), permettendo ai processi più corti di scavalcare i più lunghi. Il tempo di esecuzione stimato, come precedentemente detto può essere calcolato dal sistema operativo oppure fornito dal processo stesso
![[Pasted image 20241014234219.png|400]]

Un problema di questa politica è il fatto che potrebbe andare incontro a starvation, infatti i processi più lunghi potrebbero non andare mai in esecuzione se continuano ad arrivare processi corti. Bisogna anche tener presente che se il tempo di esecuzione stimato si rivela inesatto, il sistema operativo può abortire il processo, dunque se il tempo previsto è maggiore del tempo reale, il processo può essere terminato dal sistema operativo

### Come stimare il tempo di esecuzione?
In alcuni sistemi ci sono processi (sia batch che interattivi) che sono eseguiti svariate volte, quindi si usa il passato ($T_{i}$) per predire il futuro ($S_{i}$)
Questo può essere fatto con una media, ma vorrebbe dire per il dispatcher il doversi tenere in memoria tutti i tempi delle precedenti esecuzioni
$$
S_{n+1}=\frac{1}{n}\sum^{n}_{i=1}T_{i}
$$
Per fortuna si può fare anche in un altro modo che necessita di ricordare solamente l’ultimo tempo di esecuzione e l’ultima stima fatta
$$
S_{n+1}=\frac{1}{n}T_{n}+\frac{n+1}{n}S_{n}
$$
Questa formula può essere generalizzata chiamata *exponential averaging*
$$
S_{n+1}=\alpha\, T_{n}+(1-\alpha)S_{n}, \,\,0<\alpha<1
$$
$$
S_{n+1}=\alpha\,T_{n}+\dots+\alpha(1-\alpha)^{i}\,T_{n-1}+\dots+(1-\alpha)^{n}\,S_{1}
$$
![[Pasted image 20241015000012.png|400]]

![[Pasted image 20241015000153.png|500]]
![[Pasted image 20241015000248.png|500]]

---
## SRT (Shortest Remaining Time)
La politica shortest remaining time è simile alla SPN, ma preemptive. Questa infatti non utilizza un quanto di tempo, dunque un processo può essere bloccato solo quando ne arriva uno nuovo appena creato (o se fa un I/O bloccante). Una volta che un processo è stato interrotto viene rimesso nella coda dei ready; all’arrivo di un nuovo processo viene stimato il tempo rimanente richiesto per l’esecuzione, e viene eseguito il processo con quello più breve
![[Pasted image 20241015000902.png|400]]

---
## HRRN (Highest Response Ratio Next)
La politica highest response ratio next è una politica non preemptive. Questa richiede la conoscenza del tempo di servizio e risolve il problema della starvation. Qui viene massimizzato il seguente rapporto (oltre al tempo che ci mette viene confrontato con il tempo in cui è stato fatto aspettare, rendendo la decisione più equa):
$$
\frac{w+s}{s}=\frac{\text{tempo trascorso in attesa}+\text{tempo totale richiesto}}{\text{tempo totale richiesto}}
$$
![[Pasted image 20241016105653.png|400]]

---
## Confronto tra le varie politiche
![[Pasted image 20241016105907.png]]

---
## Scheduling tradizionale di UNIX
Le politiche di scheduling mostrate finora al giorno d’oggi non sono più semplicemente applicate spesso infatti sono utilizzati come dei blocchetti su cui poi vengono costruiti gli scheduler moderni.
Nello scheduling di UNIX è stato introdotto il concetto di **priorità combinato con il round-robin**. Quindi l’idea è che un processo resta in esecuzione per al massimo un quanto di tempo pari ad un secondo (a meno che non termini o non si blocchi). Sono inoltre presenti diverse code, a seconda della priorità, e su ogni coda viene applicato il round-robin.
Per **risolvere il problema della starvation** di questo tipo di politica però si è fatto in modo che le priorità venissero ricalcolate ogni quanto di tempo in base a quanto tempo un processo è rimasto in esecuzione (più è rimasto in esecuzione più diminuisce la priorità).

Le priorità iniziali sono basate sul tipo di processo:
- swapper (alta)
- controllo di un dispositivo di I/O a blocchi
- gestione di file
- controllo di un dispositivo di I/O a caratteri
- processi utente (basso)

### Formula di Scheduling
$$
CPU_{j}(i)=\frac{CPU_{j}(i-1)}{2}
$$
$$
P_{j}(i)=Base_{j}+\frac{CPU_{j}(i)}{2}+nice_{j}
$$
$CPU_{j}(i)$ → è una misura di quanto un processo $j$  ha usato il processore nell’intervallo $i$, con exponential averaging dei tempi passati; per i running $CPU_{j}(i)$ viene incrementato di $1$ ogni $\frac{1}{60}$ di secondo
$P_{j}(i)$ → è la priorità del processo $j$ all’inizio di $i$ (più basso è il valore, più è alta la priorità)
$Base_{j}$ → assegnato in base alle priorità iniziali del processo ($0\leq Base_{j}\leq 4$)
$nice_{j}$ → un processo può dire che il proprio valore di cortesia è maggiore di zero per auto-declassarsi, in modo tale da far passare avanti altri processi (usata prevalentemente per i processi di sistema) 

### Esempio di Scheduling su UNIX
$$
Base_{a}=Base_{b}=Base_{c}=60, \, nice_{a}=nice_{b}=nice_{c}=0
$$
![[Pasted image 20241016112748.png|center|300]]

---
## Architetture multi-processore
Esistono varie architetture multi-processore al giorno d’oggi:
- Cluster
	architettura multiprocessore con memoria non condivisa (ogni processore ha la propria RAM) e la connessione con rete locale tra di essi è superveloce
- Processori specializzati (es. ogni I/O device ha un suo processore)
- **Multi-processore e/o multi-core**
	Questi condividono la RAM (ci sta un’unica RAM che tutti i processori utilizzano) e sono controllati da un solo SO (a differenza degli altri due)

### Scheduler
Abbiamo sostanzialmente due possibilità: **assegnamento statico** e **assegnamento dinamico**. Per assegnamento si intende decidere quale processo va su quale processore
#### Assegnamento statico
Quando un processo viene creato gli viene assegnato un processore e fino alla terminazione del processo, questo viene eseguito sullo stesso processore.
La struttura dell’assegnamento statico è molto semplice, ci basterà infatti avere uno scheduler per ogni processore e dunque l’overhead sarà molto basso; lo svantaggio sta nel fatto che un processore potrebbe rimanere idle, la distribuzione dei processi infatti potrebbe non essere equa

#### Assegnamento dinamico
Per migliorare lo svantaggio dell’assegnamento statico, un processo, nel corso della sua vita potrà essere eseguito su diversi processori. Seppur ragionevole la sua realizzazione è particolarmente complessa, specie se si vuole mantenere un overhead basso.
Per semplificare la realizzazione si potrebbe decidere di fare in modo di eseguire il SO su un processore fisso, lasciando che solo i processi utente possano cambiare. Lo svantaggio però sta nel fatto che questo potrebbe diventare un bottleneck e che mentre per i processi utente il sistema potrebbe funzionare con una failure di un processore, se fallisce quello del SO no.
Una seconda possibilità è quella di eseguire il SO non su un processore fisso ma seguendo le stesse regole dei processi utente, ma che richiederebbe più overhead

---
## Scheduling in Linux
Nel corso degli anni lo scheduling di Linux è cambiato molteplici volte, quello qui presentato è uno scheduling in disuso da qualche anno.
Linux, per quanto riguarda lo scheduling, è alla ricerca di velocità di esecuzione, tramite semplicità di implementazione così da mantenere un overhead il più basso possibile. Per questo motivo in questo SO non sono presenti né long-term scheduler (anche se un suo embrione ovvero se viene creato un nuovo processo ma il sistema è già saturo), né medium-term scheduler (ci torneremo quando si parlerà di gestione della memoria).

In Linux ci sono le *runqueues* (la coda dei processi ready) e le *wait queues*
Le *wait queues* (coda dei blocked, plurale perché ci sta una coda per ogni evento) sono proprio le code in cui i processi sono messi in attesa quando fanno una richiesta che implichi l’attesa
Le *runqueues* (coda dei processi ready, plurale perché ci sta una coda per ogni priorità) sono quelle da cui pesca il dispatcher (short-term scheduler)

>[!hint] Notare
>Le *wait queues* sono condivide dai processori, invece per le *runqueues* ogni processore ha le proprie

Per quanto riguarda la politica di scheduling è sostanzialmente **derivata da quella di UNIX**: preemptive a priorità dinamica (decresce man mano che un processo viene eseguito, cresce man mano che un processo non viene eseguito) seppur con **alcune modifiche** per poter **migliorare la velocità** e per poter servire nel modo più approrpiato i processi real-time (se ci sono)