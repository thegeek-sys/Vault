---
Created: 2024-12-06
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Introduction
Le soluzione software è usabile solo per problemi di concorrenza semplici come ad esempio la mutua esclusione.
In questo caso dunque non possiamo fare affidamento alle istruzioni macchina, ma solamente all’assegnamento a variabili e simili. Come contro però si ha il fatto che tutte le operazioni sono in attesa attiva (non possono essere bloccati i processi)

---
## Prove
Facciamo dei tentativi per provare a implementare la mutua esclusione via software
### Primo tentativo
![[Pasted image 20241206174629.png]]
Questa soluzione però è applicabile solo a 2 processi (non a più processi)
#### Problemi
Una soluzione come questa risolve il problema della mutua esclusione ma con dei problemi.
Il maggiore di tutti sta nel fatto che funziona se ci sono due processi, ma se ce ne fosse uno solo (`PROCESS 1`) e `turn` fosse inizializzato a $1$ non si uscirebbe mai del processo

### Secondo tentativo
![[Pasted image 20241206175202.png]]
Il `PROCESS 0` legge la variabile di `PROCESS 1` e scrive la propria, mentre `PROCESS 1` fa il contrario

#### Problemi
In questo caso se lo scheduler interrompe `P0` immediatamente prima della modifica di `flag[0]` per passare a `P1` anche lui potrebbe entrare nella critical section e quindi si avrebbe una race condition

### Terzo tentativo
![[Pasted image 20241206175638.png]]

#### Problemi
Anche qui, se lo scheduler interrompe subito dopo aver impostato il `flag` i processi rimarrebbero bloccati nel `while` (deadlock)

### Quarto tentativo
![[Pasted image 20241206175819.png]]

#### Problemi
In questa soluzione si tenta di risolvere il problema di deadlock modificando nuovamente il `flag` dentro il `while`, ma in questo caso devo sperare che lo scheduler interrompa il processo prima della fine del `delay`. In questo caso più che deadlock, si parla di **livelock**, ovvero i processi continuano a fare qualcosa ma non di utile

---
## Algoritmo di Dekker
Nell’algoritmo di Dekker si implementano le due soluzioni, una variabile condivisa e una locale
![[Pasted image 20241206180359.png|center|580]]

Qui fin dall’inizio dichiaro di voler entrare nella sezione critica. Se il `wants_to_enter` dell’altro processo è `false` entro nella sezione critica. Nel caso in cui invece il valore è `true`, si ha una variabile `turn` condivisa. Per il `P0` se `turn` è $0$ (non tocca a me), rimetto a falso il fatto che voglio entrare e faccio un’attesa attiva finché il `turn` è 1. Una volta finita l’attesa ri-imposto il fatto che voglio entrare a `true`.

Questo algoritmo vale solo per $2$ processi estendibile a $N$ (seppur non banale, non so a cosa impostare `turn`). Garantisce inoltre la non-starvation (grazie a `turn`) e il non-deadlock (ma è busy-waiting, se ci sono delle priorità fisse, ci può essere deadlock). Non richiede alcun supporto dal SO, ma in alcune moderne architetture hanno ottimizzazioni hardware che riordano le istruzione da eseguire, nonché gli accessi in memoria, risulta dunque necessario disabilitare tali ottimizzazioni

---
## Algoritmo di Peterson
L’algoritmo di Peterson permette di risolvere lo stesso problema in maniera più semplice
![[Pasted image 20241206181525.png|center|400]]


