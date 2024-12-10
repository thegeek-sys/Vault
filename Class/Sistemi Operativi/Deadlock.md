---
Created: 2024-12-09
Class: 
Related: 
Completed:
---
---
## Introduction
Il **deadlock** (o stallo) è un blocco permanente di un certo insieme di processi che competono per delle risorse di sistema o comunicano tra loro.
Il motivo di base del deadlock è la **richiesta contemporanea delle stesse risorse** da parte di due o più processi.

![[Pasted image 20241209220045.png|center|480]]

Nonostante tutto non esiste una soluzione universale per risolvere questo problema, bisogna infatti analizzare caso per caso e risolverlo in una maniera opportuna.

---
## Joint progress diagram
Quando ci troviamo di fronte ad un deadlock tra due processi, questo può essere analizzato attraverso questo semplice diagramma

>[!info] I due processi richiedono la risorsa successiva prima di aver rilasciato quella che stanno usando

![[Pasted image 20241209220247.png]]

>[!note]
>Linee orizzontali → momento in cui `P` è in esecuzione
>Linee verticali → momento in cui `Q` è in esecuzione
>Quando intercettano un quadrante vuol dire che viene eseguita l’operazione indicata (es. Get A)
>
>5 non va in deadlock, infatti prova a richiedere una risorsa già occupata, quindi il processo viene mezzo in blocked
>Dopo 3 e 4 non si può andare da nessuna parte per come sono scritti i processi (devono richiedere l’altra risorsa prima di poter rilasciare quella che stanno usando ma è occupata)


>[!info] `P` prima di richiedere la seconda risorsa rilascia quella che sta usando, mentre `B` si comporta come prima

![[Pasted image 20241209221215.png]]

>[!note]
>Per come è strutturato questo diagramma, non ci può essere deadlock

---
## Risorse
Le risorse si distinguono in:
- **risorse riusabili**
- **risorse non riusabili**

---
## Risorse riusabili
Le risorse riusabili sono quelle risorse usabili da un solo processo alla volta, ma il fatto di **essere usate non le “consuma”**. Una volta che un processo ottiene una risorsa riusabile, prima o poi la rilascia cosicché altri processi possano usarla a loro volta
Esempio: processori, I/O channels, memoria primaria e secondaria, dispostivi, file…

Nel caso delle risorse riusabili il deadlock può avvenire solo se un processo ha una risorsa e ne richiede un’altra
### Esempio

>[!info] Esempio 1
>![[Pasted image 20241209233414.png|450]]
>
>>[!note]
>>Perform action → sezione critica
>
>Si bloccano in quanto `P` richiede `T` prima di rilasciare `D`, mentre `Q` richiede `D` prima di rilasciare `T`

>[!info] Esempio 2
>Supponiamo di trovarci in un sistema con $200\text{ KB}$ di memoria disponibili e che ci sia la seguente sequenza di richieste
>![[Pasted image 20241209233727.png|450]]
>
>Il deadlock avverrà quando uno dei due processi farà al seconda richiesta (non rilasciano la memoria)

### Condizioni per il deadlock
Il deadlock si verifica solamente se ci sono queste quattro condizioni:
- **Mutua esclusione** → solo un processo alla volta può usare una data risorsa
- **Hold-and-wait** → un processo può richiedere una risorsa mentre ne tiene già bloccate altre
- **Niente preemption** per le risorse → non si può sottrarre una risorsa ad un processo senza che quest’ultimo la rilasci
- **Attesa circolare** → esiste una catena chiusa di processi, in cui ciascun processo detiene una risorsa richiesta (invano) dal processo che lo segue nella catena

---
## Risorse non riusabili
Le risorse non riusabili sono quelle risorse che vengono **create da un processo e distrutte da un altro processo**
Esempi: interruzioni, segnali, messaggi, informazione nei buffer di I/O

Nel caso delle risorse non riusabili il deadlock può avvenire se si fa una richiesta (bloccante) per una risorsa che non è stata ancora creata, ad esempio un deadlock può avvenire su una ricezione bloccante
### Esempio

>[!info] Esempio
>![[Pasted image 20241209234502.png|450]]

### Condizioni per il deadlock
Il deadlock si verifica solamente se ci sono queste quattro condizioni:
- **Mutua esclusione** → la risorsa va a chi ne riesce a fare richiesta per primo (un messaggio arriva al primo che riceve)
- **Hold-and-wait** → si può richiedere (in modo bloccante) una risorsa che non è stata ancora creata (receive ma non ci sta la send corrispondente)
- **Niente preemption** per le risorse → non appena viene concessa, una risorsa viene distrutta
- **Attesa circolare** → esiste una catena chiusa di processi, in cui ciascun dovrebbe creare una risorsa richiesta (invano) dal processo che lo segue nella catena

---
## Grafo dell’allocazione delle risorse
Dato che il joint process diagram non è sufficiente per rappresentare le interazioni tra più processi che richiedono più risorse, si utilizza il **grafo dell’allocazione delle risorse**

Questo è un grafo diretto che rappresenta lo stato di risorse e processi (tanti pallini quante istanze di una stessa risorsa, tre pallini tre stampanti). Come nodi si utilizzano i cerchi per i processi mentre i quadrati per le risorse

In base alla direzione della freccia si determina se la risorsa è richiesta o tenuta da un processo. Mentre questo sistema risulta ok per le risorse riusabili, per le risorse consumabili non esiste mai l’”held” invece i pallini compaiono e scompaiono

![[Pasted image 20241209235006.png]]

>[!info]
>![[Pasted image 20241210001404.png|230]]
>- Mutua esclusione → sia `Ra` che `Rb` possono essere prese da un solo processo alla volta
>- Hold-and-wait → `P1` richiede `Ra` e detiene `Rb` e `P2` viceversa
>- Niente preemption → SO non può togliere le risorse
>- Attesa circolare → visivamente può essere vista da un ciclo
>
>![[Pasted image 20241210001639.png|230]]
>- Niente mutua esclusione

>[!info]
>Anche questo esempio delle macchine può essere visualizzato attraverso un grafo
>![[Pasted image 20241209220045.png|480]]
>
>![[Pasted image 20241210001822.png|480]]
>1. C’è un ciclo
>2. Nessun pallino è scoperto

---
## Possibilità ed esistenza di deadlock
### Possibilità di deadlock
Ci sta la possibilità che si presenti un deadlock quando sono verificate le prime tre condizioni:
- mutua esclusione
- richiesta di una risorsa quando se ne ha già una (hold-and-wait)
- niente preemption per le risorse
Queste infatti dipendono da come è fatto il sistema
### Esistenza di deadlock
Effettivamente è presente un deadlock quando tutte e quattro le condizioni sono verificate:
- mutua esclusione
- richiesta di una risorsa quando se ne ha già una (hold-and-wait)
- niente preemption per le risorse
- attesa circolare
L’attesa circolare invece dipende da come evolve l’esecuzione di certi processi
Si parla invece di deadlock **inevitabile** quando non è al momento presente l’attesa circolare, ma sicuramente ad un certo punto arriverà

---
## Deadlock e SO: che fare?
Esistono diverse tecniche per gestire problemi che riguardano il deadlock:
- **Prevenire** → cercando di far si che una delle 4 condizioni sia sempre falsa
- **Evitare** → decidendo di volta in volta cosa fare con l’assegnazione di risorse
- **Rilevare** → una volta che avviene il deadlock, viene notificato al SO che agisce di conseguenza
- **Ignorare** → se dei processi vanno in deadlock è colpa dell’utente, non accettabile, in generale, per processi del SO


---
## Prevenzione del deadlock
Per la prevenzione bisogna evitare che esistano contemporaneamente le 4 condizioni per un deadlock. Vediamo cosa si può evitare
- Mutua esclusione → inevitabile
- Hold-and-wait → si impone ad un processo di richiedere tutte le sue risorse in una solva volta (può essere difficile per software complessi, e si tengono risorse bloccate per un tempo troppo lungo)
- Niente preemption per le risorse → il SO può richiedere ad un processo di rilasciare le sue risorse (le dovrà richiedere in seguito); per esempio, se una sua richiesta di un’altra risorsa non è stata concessa
- Attesa circolare → si definisce un ordinamento crescente delle risorse; una risorsa viene data solo se segue quelle che il processo già detiene
Le cose più ragionevoli da risolvere sono l’hold-and-wait o l’attesa circolare

---
## Evitare il deadlock
Per evitare il deadlock bisogna decidere se l’attuale richiesta di una risorsa può portare ad un deadlock se esaudita. Ma ciò richiede la conoscenza delle richieste future, in particolare si hanno due possibilità:
- non mandare in esecuzione un processo se le sue richieste possono portare a deadlock
- non concedere una risorsa ad un processo se allocarla può portare a deadlock (algoritmo del banchiere, mando in esecuzione il processo ma non concedo la risorsa)

### Diniego delle risorse
L’algoritmo del banchiere e valido per le risorse riusabili e fa si che il programma proceda da un certo stato ad un altro stato (per stato si intende una certa situazione per l’uso delle risorse). Uno stato è **sicuro** se da essi parte almeno un cammino che non porta ad un deadlock, uno stato non sicuro se da essi partono solo cammini che portano a deadlock

### Algoritmo del banchiere
#### Strutture dati
![[Pasted image 20241210112041.png]]

>[!note] Legenda
>- $m$ → numero di diversi tipi di risorse
>- $R_{i}$ → numero di istanze dell’i-esima risorsa (effettivamente presente, non può essere modificata)
>- $V_{i}$ → numero di istanze disponibili per la i-esima risorsa
>- $C_{ij}$ → $j$ come nei precedenti si riferisce alla risorsa, $i$ invece si riferisce al numero di processi che voglio monitorare affinché non vadano in deadlock; con $C_{ij}$ ci si riferisce a quante istanze della risorsa $j$ verranno richieste (al massimo) dal processo $i$-esimo
>- $A_{ij}$ → si riferisce a quante istanze della risorsa $j$ sono state concesse al processo $i$
>- manca un vettore che indica che tipo di richiesta viene fatta

#### Determinazione dello stato sicuro
