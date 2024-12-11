---
Created: 2024-12-11
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
La particolarità dei file hash è il fatto che il file è suddiviso in **bucket** numerati da $0$ a $B-1$. Ciascun bucket è costituito da **uno o più blocchi** collegati mediante puntatori ed è organizzato come un heap

---
## Bucket
![[Pasted image 20241211001548.png|center|600]]
### Bucket directory
L’accesso ai bucket avviene attraverso la **bucket directory** che contiene $B$ elementi. L’$i$-esimo elemento contiene l’indirizzo del primo blocco dell’$i$-esimo bucket (**bucket header**)
![[Pasted image 20241211001855.png|center|600]]

---
## Funzione di hash
Dato un valore $v$ per la chiave, il numero del bucket in cui deve trovarsi un record con chiave $v$ è calcolato mediante una funzione che prende il nome di **funzione di hash**

Una funzione hash per essere “buona” deve ripartire uniformemente i record nei bucket, cioè al variare del valore della chiave deve assumere con la “stessa” probabilità uno dei valori compresi tra $0$ e $B-1$.
In generale, una funzione hash trasforma la chiave in un intero, divide questo intero per $B$ e fornisce il resto della divisione come numero del bucket in cui deve trovarsi il record con quel valore della chiave

### Esempio di funzione hash
![[Pasted image 20241211003513.png|400]]
1. trattare il valore $v$ della chiave come una sequenza di bit
2. suddividere tale sequenza in gruppi di bit di uguale lunghezza e sommare tali gruppi trattandoli come interi
![[Pasted image 20241211003614.png|300]]
$$
\textcolor{red}{9}+\textcolor{green}{7}+\textcolor{red}{10}=26
$$
3. dividere il risultato per il numero dei bucket (cioè per $B$) e prendere il resto della divisione come numero del bucket in cui deve trovarsi il record con chiave $v$
ES: se $B=3$ allora il record con chiave $v$ deve trovarsi nel bucket $2$ in quanto $26=3*8+2$

---
## Operazioni
Una qualsiasi operazione (ricerca, inserimento, cancellazione, modifica) su un file hash richiede:
- la valutazione di $h(v)$ per individuare il bucket
- esecuzione dell’operazione sul bucket che è organizzato come un heap
Poiché l’inserimento di un record viene effettuato sull’ultimo blocco del bucket è opportuno che la bucket directory contenga anche, per ogni bucket, il puntatore all’ultimo record del bucket

### Costo operazioni
Pertanto se la funzione hash distribuisce uniformemente i record nei bucket allora ogni bucket è costituito da $\frac{n}{B}$ blocchi e quindi il costo richiesto di un’operazione è approssimativamente $\frac{1}{B\text{-esimo}}$ del costo della stessa operazione se il file fosse organizzato come heap

### Inserimento
![[Pasted image 20241211003246.png|600]]

### Considerazioni
Da quanto detto appare evidente che quanti più sono i bucket tanto è più basso il costo di ogni operazione. D’altra parte limitazioni al numero di bucket derivano dalle seguenti considerazioni:
- ogni bucket deve avere almeno un blocco
- se le dimensioni della bucket directory sono tali che non può essere mantenuta in memoria principale durante l’utilizzo del file, ulteriori accessi sono necessari per leggere i blocchi dalla bucket directory

---
## Esempi
Negli esempi che seguono, così come negli esercizi di esame, a meno che non venga specificato diversamente assumeremo sempre che:
- ogni record deve essere contenuto completamente in un blocco (non possiamo avere record a cavallo di blocchi)
- i blocchi vengono allocati per intero (non possiamo allocare frazioni di blocco)

>[!example]- Esempio 1
>Supponiamo di avere un file di $250.000$ record. Ogni record occupa $300$ byte, di cui $75$ per il campo chiave. Ogni blocco contiene $1024$ byte. Un puntatore a blocco occupa $4$ byte
>
>1. Se usiamo una organizzazione hash con $1200$ bucket, quanti blocchi occorrono per la bucket directory?
>2. Quanti blocchi occorrono per i bucket, assumendo una distribuzione uniforme dei record nei bucket?
>3. Assumendo ancora che tutti i bucket contengano il numero medio di record, qual è il numero medio di accessi a blocco per ricercare un record che sia presente nel file?
>4. Quanti bucket dovremmo creare per avere invece un numero medio di accessi a blocco inferiore o al massimo uguale a $10$, assumendo comunque una distribuzione uniforme dei record nei bucket?
>
>Abbiamo i seguenti dati:
>- il file contiene $250.000$ record → $NR=250.000$
>- ogni record occupa $300$ byte → $R=300$
>- il campo chiave occupa $75$ byte → $K=75$
>- ogni blocco contiene $1024$ byte → $CB=1024$
>- un puntatore a blocco occupa $4$ byte → $P=4$
>
>>[!warning]
>>Un calcolo del tipo $\frac{NR\cdot R}{CB}$ per calcolare l’occupazione totale è sbagliato per tre motivi:
>>- avremmo record a cavallo di blocchi (se la taglia non è divisibile)
>>- avremmo blocchi a cavallo di bucket (se gli ultimi blocchi del bucket non sono riempiti per intero)
>>- mancano i puntatori al prossimo blocco nel bucket
>
>##### 1
>Indichiamo con $B$ il numero di bucket e con $BD$ il numero di blocchi per la bucket directory. La bucket directory è essenzialmente un array di puntatori indicizzato da $0$ a $B-1$
>
>Vediamo prima quanti puntatori entrano in un blocco (prendiamo la parte intera inferiore perché assumiamo che i record siano contenuti interamente nel blocco)
>$$PB=\left\lfloor  \frac{CB}{P}  \right\rfloor=\left\lfloor  \frac{1024}{4}  \right\rfloor =256 $$
>
>Ci occorreranno (prendiamo la parte intera superiore perché, non essendo stato specificato diversamente dall’esercizio, i blocchi vengono allocati interamente, e quindi la frazione di blocco va arrotondata ad un blocco intero)
>$$BD=\left\lceil  \frac{1200}{256}  \right\rceil = \lceil 4.69 \rceil= 5 $$
>
>>[!info]
>>Se viene chiesto che nella bucket directory venga memorizzato anche il puntatore all’ultimo blocco del bucket occorrere considerare coppie intere di puntatori (non possiamo spezzare in due blocchi la coppia di puntatori per un blocco)
>>$$PB=\left\lfloor  \frac{CB}{2P}  \right\rfloor $$
>
>##### 2
>Abbiamo record a lunghezza fissa, quindi supponiamo di non avere un direttorio di record all’inizio del blocco (tutto lo spazio è occupato dai dati). Serve però un puntatore per ogni blocco per linkare i blocchi dello stesso bucket. In un blocco dobbiamo quindi memorizzare il maggior numero possibile di record e in più un puntatore per un eventuale prossimo blocco nel bucket.
>Se indichiamo con $M$ il massimo numero di record memorizzabili in un blocco, avremo $M\cdot R+P\leq CB$, cioè $300M+4\leq 1024$, quindi $M\leq \frac{1020}{300}=3.4$

