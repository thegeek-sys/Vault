---
Created: 2024-11-08
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Categorie di dispositivi I/O
L’I/O è un’area assai problematica nella progettazione di sistemi operativi in quanto ci sta una grande varietà di dispositivi I/O e anche una grande varietà di applicazioni che li usano, risulta quindi difficile scrivere un SO che tanga conto di tutto

Esistono tre categorie di dispositivi:
- leggibili dall’utente
- leggibili dalla macchina
- dispositivi di comunicazione

### Dispositivi leggibili dall’utente
Sono tutti quei dispositivi usati per la comunicazione diretta con l’utente, come ad esempio:
- stampanti
- “terminali” → monitor, tastiera, mouse, ecc. (da non confondere con i terminali software)
- joystick
- ecc.

### Dispositivi leggibili dalla macchina
Sono quei dispositivi usati per la comunicazione con materiale elettronico, come ad esempio:
- dischi
- chiavi USB
- snesopri
- controllori
- attuatori

### Dispositivi per la comunicazione
Dispositivi usati per la comunicazione con dispositivi remoti, come ad esempio:
- modem
- schede Ethernet
- Wi-Fi

---
## Funzionamento (semplificato) dei dispositivi I/O
Un dispositivo di *input* prevede di essere **interrogato sula valore di una certa grandezza fisica al suo interno** (eg. per il mouse si tratta delle coordinate dell’ultimo spostamento effettuato, per la tastiera il codice Unicode dei tasti premuti, per il disco il valore dei bit che si trovano in una certa posizione al suo interno).
L’idea è che se un processo effettua una syscall `read` su un dispositivo del genere, vuole conoscere questo dato (per poterlo elaborare e fare altro)

Un dispositivo di *output* prevede di poter **cambiare il valore di una certa grandezza fisica al suo interno** (eg. per un monitor il valore RGB dei pixel, per una stampante il PDF di un file da stampare, per il disco il valore dei bit che devono sovrascrivere quelli che si trovano in una certa posizione al suo interno)
L’idea è che se un processo effettua una syscall `write` su un dispositivo del genere, vuole cambiare qualcosa (spesso l’effetto è direttamente visibile all’utente, ma in altri casi è visibile solo usando altre funzionalità di lettura, come per il disco)ù

Ci sono quindi almeno due syscall **`read`** e **`write`** (tra i dati che prendono in input queste syscall ci sta un identificativo del dispositivo su  cui si vuole eseguire l’operazione, in Linux sono i *file descriptor*)
Dunque quando arriva una di queste due syscall, il kernel comanda l’inizializzazione del trasferimento di informazioni, mette il processo in blocked e passa ad altro; la parte del kernel che gestisce un particolare dispositivo I/O è chiamata *driver* e spesso il trasferimento è fatto con DMA (viene fatto direttamente tra la RAM e il dispositivo)
A trasferimento completato, arriva l’interrupt, si termina l’operazione e il processo ritorna ready.

>[!warning]
>Questa normale esecuzione delle operazioni però potrebbe variare ad esempio in caso di fallimento dell’operazione (bad block su disco…) oppure potrebbe essere necessario fare ulteriori trasferimenti, per esempio dalla RAM dedicata al DMA a quella del processo

---
## Differenze tra dispositivi di I/O
I dispositivi di I/O possono differire sotto molti aspetti:
- data rate (frequenza di accettazione/emissione dati)
- applicazione
- difficoltà nel controllo (tastiera vs. disco…)
- unità di trasferimento dati (caratteri vs. blocchi)
- rappresentazione dei dati
- condizioni di errore

### Data rate
![[Pasted image 20241108184951.png]]

### Applicazioni
Ogni dispositivo di I/O ha una diversa applicazione ed uso
I dischi sono usati per memorizzare files, a tale scopo richiedono un software per poterli gestire. I dischi però sono anche usati per la memoria virtuale, per la quale serve un altro software apposito (nonché hardware).
Un terminate usato da un amministratore di sistema dovrebbe avere una priorità più alta

### Complessità del controllo
Per quanto riguarda la quantità di controlli necessari per i dispositivi I/O, una tastiera o un mouse richiedono un controllo molto semplice.
Ad esempio per quanto riguarda una stampante, questa è più articolata; richiedono infatti di ricevere un PDF o un PS (poi la traduzione da PDF ad azioni della stampante è ben più complessa)

Il disco è tra le cose più complicate da controllare, fortunatamente non viene fatto tutto da software, e molte cose vengono controllate da hardware dedicato

### Unità di trasferimento dati
Per trasferire i dati dal dispositivo che li genera alla memoria e viceversa ci sono due possibilità:
- trasferirli in **blocchi di byte** di lunghezza fissa → usato per la memoria secondaria (dischi, chiavi USB, CD/DVD, …)
- trasferirli come **flusso** (*stream*) **di byte** (o caratteri) → qualsiasi cosa che non sia memoria secondaria (terminali, stampanti, schede audio, dispositivi di rete…)

### Rappresentazione dei dati
I dati sono rappresentati secondo codifiche diverse a seconda del dispositivo che li genera/accetta (una vecchia tastiera potrebbe usare ASCII puro, mentre una moderna l’UNICODE)
Possono essere diversi anche i controlli di parità

### Condizioni di errore
Gli errori possono essere riparabili o non, e la loro natura varia di molto a seconda del dispositivo. Possono cambiare: nel modo in cui vengono notificati, sulle loro conseguenze (fatali, ignorabili), come possono essere gestiti

---
## Tecniche per effettuare l’I/O
Ci sono sostanzialmente quattro modalità di fare I/O

|                             | **Senza interruzioni** | **Con interruzioni**           |
| --------------------------- | ---------------------- | ------------------------------ |
| **Passando per la CPU**     | I/O programmato        | I/O guidato dalle interruzioni |
| **Direttamente in memoria** |                        | DMA                            |

### Direct Memory Access (DMA)
Quello sotto è un tipico modulo DMA
![[Pasted image 20241108193610.png|250]]

Il processore delega le operazioni di I/O al modulo DMA il quale trasferisce direttamente i dati da o verso la memoria principale. Quando l’operazione è completata il DMA genera un interrupt per il processore

---
## Evoluzione della funzione di I/O
Riassumiamo brevemente quale è stata l’evoluzione dell’I/O in ordine cronologico
1. Il processore controlla il dispositivo periferico
2. Viene aggiunto un modulo (o controllore) di I/O direttamente sul dispositivo → permettendo di fare I/O programmato senza interrupt (no multiprogrammazione), ma il processore non si deve occupare di alcuni dettagli del dispositivo stesso
3. Modulo o controllore di I/O con interrupt → migliora l’efficienza del processore, che non deve aspettare il completamento dell’operazione di I/O
4. DMA → i blocchi di dati viaggiano tra dispositivo e memoria senza usare il processore
5. Il modulo di I/O diventa un processore  separato (*I/O channel*) → il processore “principale” comanda al processore di I/O di eseguire un certo programma di I/O in memoria principale
6. Processore per l’I/O (*I/O processor* o anche *I/O channel*) → ha una sua memoria dedicata ed è usato per le comunicazioni con terminali interattivi

Nell’architettura moderna il chipset implementa le funzioni di interfaccia I/O
![[Pasted image 20241108200410.png]]

---
## Obiettivi
I sistemi operativi devono però gestire i dispositivi I/O ponendosi degli obiettivi
### Efficienza
Uno dei problemi più importanti è il fatto che la maggior parte dei dispositivi di I/O sono molto lenti rispetto alla memoria principale. Bisogna dunque sfruttare il più possibile la multiprogrammazione per evitare che questo problema di velocità diventi basso utilizzo del processore.
Nonostante ciò l’I/O potrebbe comunque non tenere il passo del processore (il numero di processi ready si riduce fino a diventare zero); come soluzione si potrebbe pensare che sia sufficiente portare altri processi sospesi in memoria principale (medium-term scheduler), ma anche questa è un’operazione di I/O

Risulta dunque necessario cercare soluzioni software dedicate, a livello di SO, per l’I/O (in particolare per il disco)

### Generalità
Nonostante siano tanti e diversi i dispositivi di I/O, bisogna comunque cercare di gestirli in maniera uniforme.
Bisogna quindi fare in modo che ci sia un’unica istruzione di `read` che in base all’argomento che gli viene dato sa come gestire suddetto dispositivo. Per questo motivo è necessario nascondere la maggior parte dei dettagli dei dispositivi di I/O nelle procedure di basso livello
Le funzionalità da offrire sono: `read`, `write`, `lock`, `unlock`, `open`, `close`, …
#### Progettazione gerarchica
Per fare ciò è necessario utilizzare una progettazione gerarchica basata su livelli, in cui ogni livello si basa sul fatto che il livello sottostante sa effettuare operazioni più primitive, fornendo servizi al livello superiore.
Inoltre ogni livello contiene funzionalità simili per complessità, tempi di esecuzione e livello di astrazione

>[!hint]
>Modificare l’implementazione di un livello non dovrebbe avere effetti sugli altri

Esistono sostanzialmente 3 macrotipi di progettazioni gerarchiche

##### Dispositivo locale
Riguarda dispositivi attaccati esternamente al computer (es. stampante, monitor, tastiera…)

![[Pasted image 20241111111852.png]]
**Logical I/O** → il dispositivo viene visto come risorsa logica (`open`, `close`, `read`, …)
**Device I/O** → trasforma le richieste logiche in sequenze di comandi di I/O
**Scheduling and Control** → esegue e controlla le sequenze di comandi, eventualmente gestendo l’accodamento
##### Dispositivo di comunicazione
Riguarda quei dispositivi che permettono la comunicazione (scheda Ethernet, WiFi, …)

![[Pasted image 20241111112117.png]]
Strutturato come prima, ma al posto del logical I/O c’è una architettura di comunicazione, tramite la quale il dispositivo viene visto come risorsa logica. A sua volta questa consiste in un certo numero di livelli (es. TCP/IP)
##### File system
Riguarda i dispositivi di archiviazione (es. HDD, SSD, CD, DVD, floppy disk, USB, …)

![[Pasted image 20241111112423.png]]
Qui il logical I/O è tipicamente diviso in tre parti
**Directory Management** → fornisce tutte le operazioni atte a gestire i file (creare, spostare, cancellare, …)
**File system** → struttura logica ed operazioni (apri, chiudi, leggi, scrivi, …)
**Organizzazione fisica** → si occupa di allocare e deallocare spazio su disco (quando ad esempio si chiede di creare un file)

---
## Buffering dell’I/O
Una delle tecniche più usate dagli SO per poter fare I/O nella maniera più efficiente è quella del **buffering** (indica una certa zona di memoria temporanea utilizzata per completare una certa operazione).

>[!info]
>Nella gestione della memoria (RAM) abbiamo visto che è presente il problema di dover spostare le pagine da memoria virtuale a disco (come effetto collaterale di usare la memoria virtuale ci sta quello di gestire l’I/O).

L’idea è quella che un processo potrebbe richiedere un I/O su una certa sua zona di memoria (gestito dal DMA), ma viene subito swappato (quindi sospeso). In questo caso si generebbe una contraddizione in quanto, per poter usare il DMA, il processo (o una sua parte) si deve trovare in memoria ma ciò non è possibile se questo deve essere swappato.
Questo problema potrebbe essere risolvibile attraverso il *frame blocking* (evitare che le pagine usate dal DMA siano swappate), tuttavia se si inizia a fare quest’operazione in maniera eccessiva limito le pagine swappabili e di conseguenza anche il numero di processi presenti in RAM.

Per risolvere questo problema è quindi risolvibile attraverso il **buffering**, ovvero, come per il prepaging, facendo in anticipo trasferimenti di input e in ritardo trasferimenti di output rispetto a quando arriva la richiesta

### Senza buffer
Se non ci fosse il buffering il SO accede al dispositivo nel momento in cui ne ha necessità
![[Pasted image 20241111134547.png]]

### Buffer singolo
Con il buffer viene messa una porzione di memoria ausiliaria nel mezzo gestita dal SO. Quindi quando arriva la richiesta di I/O viene letta e scritta nel sistema operativo, e in un secondo momento viene passata al processo utente. In questo modo è quindi prevedibile evitando il contatto diretto tra I/O e RAM
![[Pasted image 20241111134738.png]]
Lettura e scrittura nel buffer sono separate e sequenziali

---
## Buffer singolo orientato a blocchi
