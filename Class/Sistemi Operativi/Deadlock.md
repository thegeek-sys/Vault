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
