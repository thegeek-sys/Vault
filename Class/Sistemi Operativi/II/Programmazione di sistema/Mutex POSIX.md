---
Created: 2025-05-14
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## Richiami
### Flussi di esecuzione concorrente
La fonte di maggior complicazione nei sistemi operativi è costituita dall’esistenza di flussi di esecuzione concorrenti:
- in una applicazione **multithread**, ciascun thread è un flusso di esecuzione “in concorrenza” con quelli degli altri thread
- in un sistema **multiprogrammato** con prelazione, ciascun processo può essere interrotto da un altro processo
- in un sistema **multiprocessore**, diversi processi o gestori di interruzione sono in esecuzione contemporaneamente

La coerenza delle strutture di dati private di ciascun flusso di esecuzione è garantita dal meccanismo di cambio del flusso. Invece la coerenza delle strutture di dati condivise tra i vari flussi di esecuzione concorrenti non è garantita da tale meccanismo

### Race condition
Se due o più flussi di esecuzione hanno una struttura di dati in comune, la concorrenza dei flussi può determinare uno stato della struttura non coerente con la logica di ciascuno dei flussi; in particolare lo stato della memoria condivisa tra due o più flussi di esecuzione concorrenti dipende dall’ordine esatto (temporizzazione) degli accessi alla memoria stessa

Le race condition sono tra gli errori del software più insidiosi e difficili da determinare poiché:
- non sono per loro natura deterministici
- possono non apparire mai sul sistema di sviluppo
- possono non essere facilmente replicabili
- sono di difficile comprensione

>[!example] Esempio di race condition
>```c
>// flusso #1
>for (;;) {
>	crea_risorsa();
>	counter = counter+1;
>}
>
>// flusso #2
>while (counter>0) {
>	counter = counter-1;
>	consuma_risorsa();
>}
>```
>
>In questo caso `flusso #1` e `flusso #2` condividono una variabile `counter`. Però incremento e decremento non sono operazioni atomiche, ciascuna di esse infatti consiste di:
>- trasferimento del valore di `counter` in un registro (`load`)
>- operazione di incremento o decremento sul registro
>- trasferimento del valore del registro in `counter` (`store`)
>
>![[Pasted image 20250514231737.png]]
>![[Pasted image 20250514231747.png]]

### Sezione critica
Una sequenza di istruzioni di un flusso di esecuzione che accede ad una struttura di dati condivisa e che quindi non deve essere eseguita in modo concorrente ad un’altra regione critica
Per realizzare una regione critica, ovvero per avere l’accesso esclusivo ad una risorse condivisa si può usare il semaforo

#### Semaforo
Il semaforo venne inventato da E. Dijkstra ed è basato su una variabile intera contatore che viene inizializzata con valore $n>0$ e che è accessibile tramite due primitive atomiche:
- `wait` → attendi che il contatore sia positivo, poi decrementalo
- `signal` → incrementa il contatore

Il semaforo può essere contatore ($n>1$, può essere acquisito da $n$ flussi di esecuzione in modo concorrente) oppure binario ($n=1$, può essere acquisito da un solo flusso alla volta)

---
## POSIX Mutex
La **mutex** (*MUTal EXclusion device*) è utile per proteggere strutture dati condivise e realizzare sezioni critiche

Questi sono i principali comandi:

```c
int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr);
int pthread_mutex_lock(pthread_mutex_t *mutex));
int pthread_mutex_trylock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
```