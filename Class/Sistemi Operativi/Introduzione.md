---
Created: 2024-09-25
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Sistema operativo
![[Screenshot 2024-09-24 alle 15.57.16.png|center|350]]
Il sistema operativo ha il compito di  gestire le risorse hardware di un sistema computerizzato generalmente composto da:
- uno o più processori → si occupa di tutte le computazioni
- memoria primaria (RAM o memoria reale) → definita volatile, se si spegne il computer infatti se ne perde il contenuto
- dispositivi  di input/output → come ad esempio dispositivi di memoria secondaria (non volatile), dispositivi per la comunicazione (es. schede di rete) ecc.
- “bus” di sistema → mezzo per far comunicare tra loro le parti interne del computer (infatti i sistemi di input output scambiano direttamente informazioni con la ram e indirettamente con il sistema operativo attraverso la RAM)
Il suo scopo è quello di **fornire un insieme di servizi agli utenti**: sia per gli sviluppatori che per i semplici utilizzatori

---
## Registri del processore
I registri del processore si dividono in:
### Registri visibili dall’utente
Sono utilizzati soprattutto da linguaggi di basso livello e sono gli unici che possono essere **utilizzati direttamente dall’utente**. Possono contenere dati o indirizzi.
Risultano essere obbligatori per l’esecuzione di operazioni su alcuni processori, ma facoltativi per ridurre gli accessi alla memoria principale
A loro volta possono essere:
- **puntatori diretti**
- **registri indice** → per ottenere l’indirizzo effettivo, occorre aggiungere il loro contenuto ad un indirizzo di base
- **puntatori a segmento** → se la memoria è divisa in segmenti, contengono l’indirizzo di inizio di un segmento
- **puntatori a stack** → puntano alla cima di uno stack

### Registri di controllo e stato
Questi vengono usualmente **letti/modificati in modo implicito dal processore** per controllare l’uso del processore stesso **oppure da opportune istruzioni assembler** (es. `jump`) per controllare l’esecuzione dei programmi.
Nell’x86 sono considerati indirizzi di controllo anche quelli per la gestione della memoria (es. i registri che gestiscono le tabelle delle pagine)
Questi sono:
- *Program Counter* (PC) → contiene l’indirizzo di un’istruzione da prelevare dalla memoria
- *Instruction Register* (IR) → contiene l’istruzione prelevata più di recente
- *Program Status Word* (PSW) → contiene le informazioni di stato 
- *flag* → singoli bit settati dal processore come risultato di operazioni (es. risultato positivo, negativo, zero, overflow…)

### Registri interni
Questi sono usati dal processore tramite microprogrammazione e utilizzati per la comunicazione con memoria ed I/O
Questi sono:
- *Memory Address Register* (MAR) → contiene l’indirizzo della prossima operazione di lettura/scrittura
- *Memory Buffer Register* (MBR) → contiene i dati da scrivere in memoria, o fornisce lo spazio dove scrivere i dati letti dalla memoria
- *I/O address register*
- *I/O buffer register*
Per capire il funzionamento e come interagiscono MAR e MBR prendiamo questo esempio: nell’istruzione `lw $s1, 4($s2)` concettualmente, prima viene copiato l’indirizzo con l’offset di `4($s2)` in MAR, poi si utilizza il valore in MAR per accedere alla memoria e si scrive il valore letto in MBR, e infine viene copiato il contenuto di MBR in `$s1`

---
## Esecuzione di istruzioni
![[Screenshot 2024-09-25 alle 09.01.50.png|center|450]]
Un’istruzione viene eseguita in varie fasi:
1. il processore preleva (fase di fetch) le istruzioni dalla memoria principale e viene caricata nell’IR
	- il PC mantiene l’indirizzo della prossima istruzione da prelevare e viene incrementato dopo ogni prelievo. Se l’istruzione contiene una `jump`, il PC verrà ulteriormente modificato dall’istruzione stessa
3. il processore esegue ogni istruzione prelevata

### Categorie di istruzioni
Le istruzioni sono divise in base alla loro funzione
- scambio di dati tra processore e memoria
- scambio  di dati tra processore e input/output
- manipolazione di dati → include anche operazioni aritmetiche, solitamente solo con i registri, ma in alcuni processori direttamente in RAM come negli x86
- controllo → modifica del PC tramite salti condizionati o non
- operazioni riservate → (dis)abilitazione interrupt, cache, paginazione/segmentazione

### Formato istruzione ed esecuzione programma
![[Screenshot 2024-09-25 alle 09.10.14.png|450]]
![[Screenshot 2024-09-25 alle 09.11.18.png|380]]

---
## Interruzioni
Le interruzioni sono un paradigma dell’interazione hardware/software. Queste **interrompono la normale esecuzione sequenziale del programma** ed iniziano ad eseguire del software che fa parte del sistema operativo.

Le cause possono essere molteplici, e danno luogo a diverse **classi di interruzioni**:
- da programma (sincrone)
- da I/O (asincrone)
- da fallimento hardware (asincrone)
- da timer(asincrone)

Il ciclo fetch-execute cambia in questa maniera in caso di interruzioni:
![[Screenshot 2024-09-24 alle 16.34.03.png|500]]
Nel dettaglio:
![[Screenshot 2024-09-25 alle 09.44.19.png|350]]
### Interruzioni sincrone
Le uniche interruzioni sincrone sono quelle **da programma**. Come conseguenza interrompono **immediatamente** il programma. Queste nei processori Intel, sono chiamate *exception*.

Queste sono causate principalmente da:
- overflow
- divisioni per 0
- debugging
- riferimento ad indirizzo di memoria non disponibile al programma o momentaneamente non disponibile (memoria virtuale)
- tentativo di esecuzione di un’istruzione macchina errata (opcode illegale o operando non allineato)
- chiamata a *system call*

Per le interruzioni sincrone una volta che l’handler è terminato si hanno varie possibilità:
- *faults* → errore correggibile, viene rieseguita la stessa istruzione
- *aborts* → errore non correggibile, si esegue software collegato all'errore
- *traps* (per debugging) e *system calls* → si continua dall’istruzione successiva

### Interruzioni asincrone
Questo tipo di interruzioni vengono tipicamente sollevate (molto) **dopo** l’istruzione che le ha causate (addirittura, alcune non sono neanche causate dall’esecuzione di istruzioni). Queste nei processori Intel, sono chiamate *interrupt*.

Le cause possono essere molteplici:
- interruzioni da **input/output**
	Queste sono generate dal controllore di un dispositivo I/O e vengono generati perché generalmente questi sono più lenti del processore. Per questo motivo il processore manda un comando al dispositivo di I/O (es. leggere/scrivere un file) e, per evitare che il processore aspetti per l’esito dell’operazione, nel mentre **continua ad eseguire altre operazioni**.
	Quando il dispositivo termina l’operazione, genera un’interruzione che segnala alla CPU di **fermarsi momentaneamente per gestire** la richiesta del dispositivo
- interruzioni da **fallimento hardware**
	Come ad esempio un’improvvisa mancanza di potenza o un errore di parità nella memoria
- interruzioni da **comunicazione tra CPU** (nei sistemi in cui ce n’è più di una)
- interruzioni da **timer**
	Generate da un timer interno al processore. Queste permettono al sistema operativo di eseguire alcune operazioni ad intervalli regolati

Per questo tipo di interruzioni, una volta che l’handler è terminato, **si riprende dall’istruzione subito successiva** a quella dove si è verificata l’interruzione (ovviamente solo se la computazione non è stata completamente abortita, anche se in alcuni casi ci potrebbe essere un process switch)