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

### Risorse riusabili
Le risorse riusabili sono quelle risorse usabili da un solo processo alla volta, ma il fatto di **essere usate non le “consuma”**. Una volta che un processo ottiene una risorsa riusabile, prima o poi la rilascia cosicché altri processi possano usarla a loro volta
Esempio: processori, I/O channels, memoria primaria e secondaria, dispostivi, file…

Nel caso delle risorse riusabili il deadlock può avvenire solo se un processo ha una risorsa e ne richiede un’altra
#### Esempio
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

### Risorse non riusabili
Le risorse non riusabili sono quelle risorse che vengono **create da un processo e distrutte da un altro processo**
Esempi: interruzioni, segnali, messaggi, informazione nei buffer di I/O