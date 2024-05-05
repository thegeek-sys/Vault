---
Created: 2024-05-02
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Fino ad ora abbiamo immaginato ogni istruzione come divisa in cinque fasi:
- Instruction fetch (IF): **memoria istruzioni (e aggiornamento PC)**
- Instruction Decode (ID): **blocco registri (e CU)**
- Execute (EXE): **ALU**
- Memory access (MEM): **memoria dati**
- Write Back (WB): **blocco registri

Queste cinque fasi vengono però svolte in sequenza poiché non è possibile eseguire la fase successiva senza il risultato della precedente, rendendo quindi le altre 4 fasi temporaneamente inutilizzabili.

Per rendere il più efficiente possibile la nostra architettura, dunque, possiamo scomporre l’esecuzione di un’istruzione in una catena di montaggio (**Pipeline**), dove ogni fase svolge il compito ad essa assegnatogli per poi **passare il risultato alla fase successiva**, procedendo ad elaborare già la **stessa fase** dell’istruzione successiva:
![[Screenshot 2024-05-02 alle 16.41.46.png]]

---
## Incremento della velocità
Utilizzando la pipeline possiamo quindi **ridurre il periodo di clock**, dalla durata massima di un’istruzione, alla **durata massima di una fase** (se fosse minore si accumulerebbero le istruzioni e dunque non potrebbero essere lette ed eseguite in modo corretto).
Nel caso ideale in cui ogni fase impiega lo stesso tempo, dunque la velocità sarà cinque volte più veloce rispetto all’esecuzione sequenziale delle fasi.

Immaginiamo che le 5 fasi abbiano le seguenti tempistiche:
- *Instruction Fetch* → 200ps  
- *Instruction Decode* → 100ps
- *Instruction Execute* → 200ps
- *Memory Access* → 200ps
- *Write Back* → 100ps
Normalmente, per poter eseguire l’istruzione più lenta possibile, ossia richiedente il completamento di tutte e 5 le fasi (ad esempio Load Word), sarebbe necessario utilizzare un periodo di clock pari a 800ps.
Tramite l’implementazione della pipeline, invece, tale periodo può essere ridotto a quello della **fase più lenta**, ossia $200\text{ps}$, aumentando quindi la velocità dell’architettura.

### Lettura e Scrittura dal blocco Registri
E’ però da notare che la fase di **lettura** (ID) che quella di **scrittura** (WB) lavorano sui registri, impiegando un tempo notevolmente minore (la metà in questo esempio, 100ps) rispetto a tutte le altre fasi dell’istruzione. Possiamo per questo eseguire ID e W nello stesso clock (periodo 200ps), suddividendolo in lettura e scrittura.
![[Screenshot 2024-05-02 alle 18.42.47.png]]

In particolar modo possiamo eseguire la **scrittura durante il rising edge** e la **lettura durante il falling edge** in modo tale da non perdere un intero ciclo di clock quando le due istruzioni si sovrappongono
![[Screenshot 2024-05-02 alle 18.44.10.png|450]]

---
## Criticità nell’esecuzione
L’implementazione della pipeline all’interno dell’architettura comporta anche la nascita di alcune criticità (**hazard**) dovute alla suddivisione in fasi delle istruzioni

Immaginiamo il caso in cui l’istruzione 1 **modifichi** il valore di un registro e l’istruzione 2 legga il valore di tale registro. Per via della suddivisione in fasi, durante la **fase di ID dell’istruzione 2** non è ancora stata eseguita la **fase di WB dell’istruzione 1**, generando quindi una **situazione critica** in cui il dato del registro non sia ancora stato modificato. Di conseguenza, l’istruzione 2 **leggerà il dato non ancora aggiornato**.

Gli hazard possono essere di tre tipi:
- **Structural hazard** → le risorse hardware non sono sufficienti (memoria dati e memoria istruzioni condivise in una singola memoria, risolto in fase di design)
- **Data hazard** → il dato necessario non è ancora pronto
- **Control hazard** → la presenza di un salto cambia il flusso di esecuzione delle istruzioni

---
## Data Hazard
Immaginiamo di avere:
```asm
addi $s0,$s1,5
sub $s2,$s0,$t0
```

Per via della scomposizione in fasi dell’esecuzione dell’istruzione, si verifica un *Data Hazard* sul registro `$s0`, il cui valore aggiornato non risulta ancora scritto, poiché la fase di WB dell’istruzione modificante non è ancora stato portato a termine. Di conseguenza, la fase di ID dell’istruzione direttamente successiva **leggerà il valore errato**

|                   | 1° CC | 2° CC | 3° CC  | 4° CC | 5° CC  | 6° CC |
|:----------------- |:-----:|:-----:|:------:|:-----:|:------:|:-----:|
| `addi $s0,$s1,5`  |  IF   |  ID   |  EXE   |  MEM  | **WB** |       |
| `sub $s2,$s0,$t0` |       |  IF   | **ID** |  EXE  |  MEM   |  WB   |
La fase di WB della 1° istruzione e la fase di ID della seconda risultano sfalzate di 2cc, generando una lettura sbagliata del dato

Per risolvere la criticità, dunque, possiamo **allineare le fasi di WB e ID** delle due istruzioni introducendo due stalli nella pipeline, ossia un’istruzione fantoccio, detta **NOP** (*No-Operation*), che funga da "rallentamento" nel caricamento della pipeline, risolvendo il data hazard.

|                   | 1° CC | 3° CC | 2° CC | 4° CC | 5° CC  | 6° CC | 7° CC | 8° CC |
| :---------------- | :---: | :---: | :---: | :---: | :----: | :---: | ----- | ----- |
| `addi $s0,$s1,5`  |  IF   |  EXE  |  ID   |  MEM  | **WB** |       |       |       |
| `sub $s2,$s0,$t0` |       |   →   |   →   |  IF   | **ID** |  EXE  | MEM   | WB    |

### Propagazione (Bypassing/Forwarding)
Tuttavia, in alcuni casi l’informazione aggiornata necessaria è **già presente** all’interno di uno dei **banchi di registri precedenti al WB**. Immaginiamo quindi che nell’architettura sia presente una **"scorciatoia"** in grado di sovrascrivere il dato errato con il dato aggiornato, senza dover attendere la fase di WB.
![[Screenshot 2024-05-02 alle 19.17.01.png]]

L’uso di questa scorciatoia rimuove la necessità di dover inserire due stalli all’interno della pipeline, velocizzando l’esecuzione del programma. Tale tecnica di propagazione del dato viene detta ***Forwarding*** (o Bypassing)

### Bubble (bolla)
Nel caso in cui la fase che necessità il dato aggiornato si trova prima della fase in cui viene aggiornato il dato, sarà comunque necessario **introdurre qualche stallo**, in modo da rallentare l’esecuzione in attesa che il dato venga generato, per poi leggerlo subito dopo attraverso il forwarding.

Nel seguente esempio il dato aggiornato viene generato in **fase di accesso alla memoria**, tuttavia, durante tale fase MEM, viene svolta in contemporanea la fase di EXE dell’istruzione successiva, la quale necessiterebbe del dato aggiornato. **Poiché il dato non può essere contemporaneamente generato e propagato** tramite il forwarding, è necessario introdurre **almeno uno stallo**.
![[Screenshot 2024-05-02 alle 19.24.56.png]]

![[Screenshot 2024-05-02 alle 19.27.01.png|470]]

### Riordinamento delle istruzioni
Spesso è possibile risolvere Data Hazard semplicemente riordinando le istruzioni che devono essere eseguite in modo tale da non dover aggiungere alcun tipo di stallo

![[Screenshot 2024-05-05 alle 17.41.24.png]]

>[!warning]
>Ciò è possibile purché nel programma venga mantenuta sempre la stessa semantica

---
## Control Hazard
Come sappiamo, l’esecuzione di un programma prevede la **lettura sequenziale** delle operazioni da svolgere, le quali vengono caricate man mano all’interno della pipeline.

Tuttavia, ciò può risultare problematico nel caso in cui debba essere effettuato un **salto**, poiché possono verificarsi due casistiche:
- Il salto **non viene eseguito**, non modificando il flusso di esecuzione delle istruzioni, procedendo normalmente con l’istruzione sottostante al branch appena validato.
	In tal caso, non sarà necessario andare a lavorare sulla pipeline, poiché tale istruzione risulterà già caricata in essa per via della lettura sequenziale.
	
- Il salto **viene eseguito**, modificando il flusso di istruzioni, procedendo normalmente con l’istruzione associata all’etichetta data al branch appena validato.
	In tal caso, sarà necessario andare a lavorare sulla pipeline, poiché sarà **necessario rimpiazzare l’istruzione attualmente caricata** nella pipeline (ossia quella sottostante al branch) con l’istruzione su cui viene effettuato il salto.

Prendiamo il seguente codice
```asm
.text
main:
	li $s0,4
	li $s1,7
	
	...
	altre istruzioni
	...
	
	beq $s0,$s1,lab
	subi $s0,$s0,4

lab:
	addi $s0,$s04
```

![[Screenshot 2024-05-05 alle 17.53.32.png|300]]

Però nel caso in cui il controllo del branch sia **vero** è necessario scartare l’istruzione precedentemente caricata, in modo tale da poter caricare invece l’istruzione corretta su cui viene effettuato il salto
![[Screenshot 2024-05-05 alle 17.54.35.png|550]]

Tale politica di salto, ossia la modalità di gestione dei salti, viene detta politica **Branch not Taken** e si basa sull’assunzione che il salto venga sempre **considerato come falso**, dunque non preso, caricando quindi l’istruzione direttamente successiva, per poi essere eventualmente scartata e rimpiazzata nel caso in cui tale assunzione si verifichi sbagliata, effettuando quindi il salto.

Un’ulteriore politica di salto utilizzabile è la politica **Branch Taken** basata sull’assunzione che il salto venga sempre **considerato come vero**, dunque preso, caricando quindi l’istruzione su cui viene effettuato il salto, per poi essere eventualmente scartata e rimpiazzata nel caso in cui tale assunzione si verifichi sbagliata, annullando quindi il salto.