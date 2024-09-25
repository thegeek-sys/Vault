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