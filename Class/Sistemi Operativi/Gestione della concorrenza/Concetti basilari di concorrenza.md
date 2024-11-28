---
Created: 2024-11-25
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Processi multipli
Per i SO moderni, è essenziale supportare più processi in esecuzione in uno di questi tre modi:
- multiprogrammazione
- multiprocessamento (*multiprocessing*)
- computazione distribuita (cluster)

Il grosso problema ora da affrontare è la **concorrenza**, ovvero gestire il modo con cui questi processi interagiscono
### Multiprogrammazione
Se c’è un solo processore, i processi si alternano nel suo uso (*interleaving*)
![[Pasted image 20241125223829.png|480]]

### Multiprocessing
Se c’è più di un processore, i processi si alternano (*interleaving*) nell’uso di un processore, e possono sovrapporsi nell’uso dei vari processori (*overlapping*)
![[Pasted image 20241125224056.png|480]]

---
## Concorrenza
La concorrenza si manifesta nelle seguenti occasioni:
- applicazioni multiple → c’è condivisione del tempo di calcolo (a carico del SO, le altre no)
- applicazioni strutturate per essere parallele → perché generano altri processi o perché sono organizzate in thread
- struttura del sistema operativo → gli stessi SO operativi sono costituiti da svariati processi o thread in esecuzione parallela

### Difficoltà
La difficoltà principale quando si parla di concorrenza è il fatto che **non si può fare alcuna assunzione sul comportamento dei processi** né sul comportamento dello scheduler

Un altro problema lo si ha nella **gestione delle risorse** (es. stampante), ovvero quando più processi tentano di accedere ad una risorsa ma che può essere acceduta da un solo processo (ma anche più semplice come quando 2 thread che accedono alla stessa variabile globale)

Inoltre si ha il problema della **gestione dell’allocazione delle risorse condivise** (decidere se dare o no una risorsa condivisa ad un processo), infatti la concorrenza fa in modo che non esista una gestione ottima del carico in quanto si potrebbe incorrere in race condition (es. processo potrebbe richiedere un I/O e poi essere rimesso ready prima di usarlo: quell’I/O va considerato locked oppure no?)

Risulta infine **difficile tracciare gli errori di programmazione**, quando viene violata la mutua esclusione ad esempio potrebbe essere un caso molto particolare dovuto alle scelte dello scheduler (tentando di ricreare la situazione che ha generato inizialmente l’errore potrebbe capitare che l’errore non avvenga)

---
## Terminologia
Per poter affrontare il problema della concorrenza al meglio è necessario imparare della terminologia:
- **Operazione atomica** → sequenza indivisibile di programmi; il disparcher non può interrompere queste operazioni fino alla loro terminazione (nessun altro processo può vedere uno stato intermedio della sequenza o interrompere la sequenza)
- **Sezione critica** → una parte del codice di un processo in cui viene fatto un accesso ad una risorsa condivisa
- **Mutua esclusione** → questo è il problema principe quando si parla di concorrenza (gli altri sono più o meno collegati); avviene quando due processi provano ad accedere ad una risorsa condivisa, ma che è fatta in modo che solo un processo alla volta la può usare
- **Corsa critica** (*race condition*) → caso in cui la mutua esclusione viene violata (a causa di errori di programmazione)
- **Stallo** (*deadlock*) → situazione nella quale due o più processi non possono procedere con la prossima istruzione (tutti i processi della catena aspettano un processo all’interno della catena)
- **Stallo attivo** (*livelock*) → situazione nella quale due o più processi cambiano continuamente il proprio stato, l’uno in risposta all’altro, senza fare alcunché di “utile”
- **Morte per fame** (*starvation*) → un processo, pur essendo ready, non viene mai scelto dallo scheduler

---
## Esempio facile
Immaginiamo di avere questa procedura
```c
/* chin e chout sono globali */
void echo() {
	chin = getchar(); // prende char in input
	chout = chin;
	putchar(chout); // stampa a schermo il char
}
```

### Esempio su un processore
Supponiamo che ci siano due processi che tentano di eseguire la stessa procedura su un processore
```c
Process P1                    Process P2
    .                             .
chin = getchar();                 .
	.                         chin = getchar();
chout = chin;                     .
	.                         chout = chin;
putchar(chout);                   .
	.                         putchar(chout);
	.                             .
```

In questo caso avremmo in output lo stesso carattere nonostante ai due processi siano stati dati due input diversi (in P1 viene scritto il valore dato in input a P2)

### Esempio su più processori
Non necessariamente l’avere più processori risolverebbe in automatico il problema
```c
Process P1                    Process P2
    .                             .
chin = getchar();                 .
	.                         chin = getchar();
chout = chin;                 chout = chin;
putchar(chout);                   .
	.                         putchar(chout);
	.                             .
```

In questo caso infatti avremmo nuovamente lo stesso problema

---
## Restrizione all’accesso singolo
Risolvere il problema dell’esempio precedente risulta essere particolarmente semplice. La soluzione infatti sta nel permettere l’esecuzione della funzione `echo` ad un solo processo alla volta (può essere richiesta da tutti i processi ma solo uno alla volta la può eseguire).
Questo viene chiamato rendere **atomica** una funzione