---
Created: 2024-05-05
Class: "[[Architettura degli elaboratori]]"
Related:
  - "[[Pipeline]]"
Completed:
---
---
## Introduction
Adesso che è chiaro cosa sia una pipeline possiamo iniziare a pensare a come implementarla all’interno di una CPU MIPS.
Per farlo e permettere quindi il **forwarding** (operazione cardine all’interno della pipeline) è necessario integrare dei registri tra le varie unità funzionali, in modo tale da poter recuperare un dato se necessario
![[registri pipeline_1.jpeg]]
### Esempio
Prendiamo in analisi il seguente codice:
![[FFA40860-4F15-464E-9035-C9ED6C4D2CA9_1_201_a.jpeg]]

In questo caso avremmo un problema durante il Write Back dell’istruzione `lw`. Questo avviene in quando se guardiamo le fasi dell’esecuzione risulta facile notare come, nel momento in cui l’istruzione $\enclose{circle}{1}$ esegue il WB l’istruzione $\enclose{circle}{4}$ ha già eseguito IF e dunque all’interno del blocco dei registri sono già pronti i registri del `sw` per essere letti e scritti. Dunque risulterebbe che il registro di destinazione di `lw` al posto di essere `$t4` risulta essere `$t6`
![[FFA40860-4F15-464E-9035-C9ED6C4D2CA9_1_201_a 1.jpeg|550]]

Per questo motivo tutte le informazioni ed i segnali di controllo devono essere nel registro precedente della pipeline
![[Screenshot 2024-05-05 alle 19.06.38.png]]

---
## Con logica dei salti (beq)
Aggiungendo la logica dei salti (beq) e integrandola con i registri già esistenti, ne approfitto per spostare tutti i controlli dopo ID in modo tale da aver la necessità di effettuare controlli solo durante le ultime tre fasi dell’istruzione (rimane solamente `RegWrite` che però non mi crea problemi in quanto viene attivato solamente durante il WB).
Si noti che ora serve il campo `funz` (codice funzione) su 6 bit dell’istruzione, nello stadio EX dove viene utilizzato come ingresso del controllore della ALU; occorre quindi salvare anche questi bit nel registro di pipeline ID/EX
![[Screenshot 2024-05-06 alle 16.40.11.png]]

---
## Segnali di controllo della CU
Possiamo quindi dividere i segnali di controllo nelle fasi esecutive in 3 gruppi:
- fase EXE
- fase MEM
- fase WB
![[Screenshot 2024-05-06 alle 16.50.34.png]]

Così si propagano i segnali di controllo all’interno della CPU
![[Screenshot 2024-05-06 alle 16.56.42.png]]

CPU nella sua totalità completa di segnali di controllo uscenti dalla porzione dei registri di pipeline dedicata a memorizzare le informazioni di controllo
![[Screenshot 2024-05-06 alle 16.58.29.png]]

---
## Scoprire un data hazard in EXE
Immaginiamo di avere questa sequenza di istruzioni:
![[Screenshot 2024-05-08 alle 16.38.51.png]]

In questo caso, nonostante tutte le istruzioni utilizzino il registro `$2` le uniche istruzioni che avranno il risultato corretto di `sub` saranno `add` e `sw` in quanto le altre due leggerebbero solamente il valore precedentemente immagazinato in `$2`.

Ma risulta facile capire come il risultato dell’istruzione `sub` sia disponibile già al termine della fase EX.
Ma quando hanno realmente bisogno l’istruzione `and` e `or` del dato? Hanno bisogno del dato di `$2` solamente all’inizio della fase EX (cc 4 e 5), di conseguenza è possibile eseguire questo frammento di codice senza stalli semplicemente **propagando** il dato a qualsiasi unità lo richieda non appena è disponibile.

Questo veloce esempio ci fa subito capire quali siano le casistiche che ci portano ad un data hazard in EXE:
1. $\text{EX/MEM.RegistroRd}=\text{ID/EX.RegistroRs}$
2. $\text{EX/MEM.RegistroRd}=\text{ID/EX.RegistroRt}$
3. $\text{MEM/WB.RegistroRd}=\text{ID/EX.RegistroRs}$
4. $\text{MEM/WB.RegistroRd}=\text{ID/EX.RegistroRt}$

Il primo hazard generato dall’esempio è quello tra `sub $2,$1,$3` e `and $12,$2,$5`. Tale hazard può essere rilevato quando l’istruzione `and` si trova allo stadio EX e l’istruzione precedente si trova nello stadio MEM; si tratta quindi di un hazard di tipo 1 $\text{EX/MEM.RegistroRd}=\text{ID/EX.RegistroRs}=\$2$

Ma dato che **non tutte le istruzioni scrivono il risultato nel register file**, questa strategia non è precise e potrebbero esserci casi in cui viene propagato un dato anche se non è necessario.
Una possibile soluzione consiste nel verificare se il segnale **`RegWrite` è attivo** nella porzione dei registri della pipeline $\text{EX/MEM}$ e $\text{MEM/WB}$. E’ da ricordare inoltre che l’architettura MIPS richiede che il registro `$0` contenga sempre 0. Se dunque nella pipeline si abbia `$0` come registro di destinazione, si cerca di evitare che il risultato dell’operazione sia propagato in avanti.

Se possiamo quindi prendere gli input della ALU non solo dal registro $\text{ID/EX}$ ma anche dagli altri registri della pipeline allora possiamo propagare i dati corretti, ciò lo faccio attraverso dei MUX sugli ingressi della ALU.

Dunque si ha un **data-hazard in EXE** quando:
![[Screenshot 2024-05-08 alle 18.04.18.png]]

> [!hint]-
> Aggiungo anche `MemRead==0` poiché nel caso di operazioni immediate:
> ![[Screenshot 2024-05-08 alle 18.09.08.png]]
> In questo caso infatti quella che dovrebbe essere la parte dell’istruzione immediata di `lw`, vista come `rd`, risulta essere uguale a `rs` dell’`add` facendoci ricadere nel caso 1a (non considerando MemRead)


Viene dunque implementata in questo modo la propagazione su EXE
![[Screenshot 2024-05-08 alle 18.16.26.png|580]]
con i seguenti segnali di controllo

| Segnale multiplexer | Sorgente | Spiegazione                                                                                             |
| ------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| PropagaA = 00       | ID/EX    | Il primo operando della ALU proviene dal register file                                                  |
| PropagaA = 10       | EX/MEM   | Il primo operando della ALU viene propagato dal risultato della ALU nel ciclo di clock precedente       |
| PropagaA = 01       | MEM/WB   | Il primo operando della ALU viene propagato dalla memoria dati o da un precedente risultato della ALU   |
| PropagaB = 00       | ID/EX    | Il secondo operando della ALU proviene dal register file                                                |
| PropagaB = 10       | EX/MEM   | Il secondo operando della ALU viene propagato dal risultato della ALU nel ciclo di clock precedente     |
| PropagaB = 01       | MEM/WB   | Il secondo operando della ALU viene propagato dalla memoria dati o da un precedente risultato della ALU |

### Realizzare forwarding in per data hazard in EXE
**Modifiche al data path** → inserire MUX prima della ALU per selezionare i 3 casi:
- *non c’è forwarding*
	il valore per la ALU viene dal registro **ID/EX** della pipeline
- *forwarding dall’istruzione precedente*
	il valore per la ALU viene dal registro **EX/MEM** della precedente
- *forwarding da 2 istruzioni precedenti*
	il valore per la ALU viene dal registro **MEM/WB** della precedente

Questo vale sia per il primo che per il secondo argomento della ALU

>[!warning] E' possibile realizzare il forwarding anche:
>- nella fase ID → necessario solo se la `beq` viene anticipata in ID
>- nella fase MEM → necessario SOLO se `lw $rd...`, è subito seguita da `sw $rd...`, se `sw` è preceduta da tipo R il forwarding avviene in fase EXE

---
## Scoprire data hazard in MEM (lw/sw)
Un data hazard in MEM, come visto nel callout precedente, lo si ha solamente quando viene fatto in sequenza un `lw` e uno `sw` su uno stesso registro `$rd`.
Questo viene fatto quando vogliamo spostare un valore dalla memoria e posizionarlo in un altro valore della memoria. Una sorta di swap di valori in memoria.

![[Screenshot 2024-05-08 alle 18.51.27.png]]

Di un data hazard di questo tipo ce ne accorgiamo quando una operazione precedente tenta di scrivere un dato in memoria nel registro in $\text{MEM/WB}$ e l’istruzione precedente sta invece tentando di scrivere in memoria in $\text{EXE/MEM}$

Si ha quindi:
![[Screenshot 2024-05-08 alle 18.57.20.png|600]]
![[Screenshot 2024-05-08 alle 18.58.03.png]]

### Realizzare forwarding in per data hazard in MEM
![[Screenshot 2024-05-08 alle 18.59.06.png|470]]

---
## Stallo dell’istruzione
Talvolta però risulta necessario che l’istruzione debba attendere che sia pronto il dato prima di poter effettuare un forwarding. 

Guardiamo il seguente esempio:
![[Screenshot 2024-05-08 alle 19.05.47.png]]

Come possiamo notare in questo caso il risultato della prima istruzione `lw` non è pronto prima della fase MEM il che vuol dire che bisognerà mettere aggiungere uno stallo. Se l’istruzione nello stadio ID viene messa in stallo, anche l’istruzione nello stadio IF deve essere messa in stallo, altrimenti l’istruzione prelevata nella fase di fetch precedente andrebbe persa. Per impedire a queste due istruzioni di procedere basta impedire la **modifica del PC e del registro di pipeline IF/ID**.
Questa “pausa“ viene eseguita attraverso l’ausilio di istruzioni che non producono effetti dette **nop** (not operation). Tra le nop troviamo l’impostare tutti i campi a 0

Dunque se identifichiamo l’hazard nello stadio ID, possiamo inserire nella pipeline una “bolla” mettendo tutti i campi a zero.

Nell’esempio riportato, l’hazard costringe le istruzioni `and` e `or` a ripetere nel quarto ciclo di clock ciò che avevano fatto nel terzo ciclo
![[Screenshot 2024-05-08 alle 19.20.51.png]]

### Eseguire uno stallo
In sintesi per aggiungere uno stallo (nella fase ID), dobbiamo:
- **annullare l'istruzione che deve attendere** (bolla); il che implica azzerare segnali di controllo `MemWrite` e `RegWrite` nonché $\text{IF/ID.Istruzione}$
- **rileggere la stessa istruzione** nuovamente affinché possa essere rieseguita un ciclo di clock dopo, devo quindi **impedire che il PC si aggiorni**

### La CPU fino a questo momento
![[Screenshot 2024-05-08 alle 19.26.03.png]]

---
## Control Hazard
Oltre ad hazard che coinvolgono le operazioni aritmetiche ci sono anche hazard che coinvolgono i **salti condizionati**.

Uno dei primi problemi da affrontare con i salti è il fatto che richiederebbe troppo tempo attendere che la logica del processore calcoli se il salto è da effettuare o meno, viene per questo introdotta la politica del **branch not taken** (o branch taken).
Questo consiste nel considerare sempre di non dover effettuare il salto (quindi di procedere con l’esecuzione sequenziale delle istruzioni). Se invece risultasse che il salto doveva essere eseguito, le istruzioni che si trovano nella fase IF, ID ed EXE devono essere scartate e l’esecuzione deve riprendere a partire dall’istruzione contenuta all’indirizzo di destinazione del salto (ricorda che non si tratta di un vero e proprio salto all’indirizzo, è  infatti un salto relativo).
Per **scartare le istruzioni** bisogna semplicemente cambiare il valore dei segnali di controllo forzandoli a 0, analogamente a come si è fatto per ottenere lo stallo nei data hazard). La differenza sta nel fatto che in questo caso non bisogna solamente “mettere in pausa” il processore, bensì bisogna proprio **cancellare** le istruzioni, sono per questo costretto ad effettuare un *flush*, che consiste nello svuotamento delle fasi IF, ID e EXE


Un’ulteriore miglioria che si potrebbe apportare al nostro processore consiste nell’**anticipare la decisione sul salto** in modo tale da poter scartare un minor numero di istruzioni.
Per farlo devo anticipare due azioni:
1. il **calcolo dell’indirizzo di destinazione del salto**
2. la **valutazione se saltare o meno**
Per il calcolo dell’indirizzo di destinazione, dato che il contenuto del PC è già disponibile, mi è sufficiente spostare il sommatore che calcola l’indirizzo di salto dalla fase EXE alla fase ID
Per quanto riguarda la decisione sul salto, nel caso di branch equal, si deve confrontare l’uguaglianza del contenuto dei due registri letti durante la fase di ID (lo si può fare mettendo in uno XOR i bit corrispondenti del contenuto dei registri, e mandare tutti i risultati in un OR, se il risultato sarà un 1 i registri sono diversi, altrimenti sono uguali).

>[!warning]
>Dobbiamo comunque tenere a mente che anticipare questa decisione ad ID implica l’aggiunta di altra di propagazione e di rilevazione degli hazard, dato che la decisione sul salto dai dati contenuti dentro la pipeline, dovremo assicurare che la branch funzioni correttamente anche con questa ottimizzazione.

Questa miglioria ci permette di scartare una sola istruzione invece di due (cioè quella che si trova nella fase di fetch). L’eliminazione viene effettuata da un segnale di *flush* che azzera la parte del registro di pipeline $\text{IF/ID}$ che contiene l’istruzione, rendendo l’operazione una cosiddetta `nop` (istruzione che non fa nulla)