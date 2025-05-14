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

### $\verb|pthread_mutex_init|$

```c
int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr);
```

Il comando `pthread_mutex_init` inizializza un mutex e imposta gli attributi a `mutexattr`
Gli attributi determinano il comportamento del semaforo quando il thread invoca un lock/unlock più volte consecutivamente

Se `mutexattr=NULL` il tipo di `mutexattr` viene impostato a fast. Possibili tipi di `pthread_mutexattr_t`:
- `PTHREAD_MUTEX_NORMAL` (fast) → un lock blocca il thread finche’ il lock precedente non è rilasciato (può creare stallo); unlock rilascia il semaforo e ritorna subito
- `PTHREAD_MUTEX_RECURSIVE` → permette allo stesso thread di mettere più lock (un contatore tiene conto del numero di lock messi); unlock decrementa il contatore ma non rilascia il semaforo finchè il contatore non è $0$
- `PTHREAD_MUTEX_ERRORCHECK` → genera un errore in caso il thread cerchi di mettere un lock su di un mutex sul quale già di detiene un lock; unlock ritorna errore se il thread non aveva fatto una lock precedentemente

Il tipo può essere impostato tramite la funzione di libreria `pthread_mutexattr_settype`

### $\verb|pthread_mutex_lock|$

```c
int pthread_mutex_lock(pthread_mutex_t *mutex));
```

Acquisisce (blocca) il mutex. Se il mutex è già bloccato da un altro thread, il chiamante si blocca in attesa
Ritorna $0$ se il mutex è libero e viene acquisito

### $\verb|pthread_mutex_trylock|$

```c
int pthread_mutex_trylock(pthread_mutex_t *mutex);
```

Tenta di acquisire il mutex, ma non blocca il thread chiamante
Se il mutex è libero, allora lo acquisisce e ritorna $0$, se invece è occupato ritorna il valore `EBUSY`

### $\verb|pthread_mutex_unlock|$

```c
int pthread_mutex_unlock(pthread_mutex_t *mutex);
```

Rilascia il mutex precedentemente acquisito
Ritorna $0$ in caso di esito positivo, altrimenti viene restituito un numero per indicare l’errore

### $\verb|pthread_mutex_destroy|$

```c
int pthread_mutex_destroy(pthread_mutex_t *mutex);
```

Distrugge un mutex, liberando risorse
Ritorna $0$ in caso di esito positivo, altrimenti viene restituito $-1$ e viene impostato `errno`

---
## Sincronizzazione thread
### Barriera
La **barriera** è un metodo di sincronizzazione tra processi o thread. Fa si che un set di processo o thread possa continuare il proprio flusso di esecuzione solo se tutti hanno raggiunto la barriera

```c
int pthread_barrier_init(pthread_barrier_t * restrict barrier, const pthread_barrierattr_t * restrict attr, unsigned int count);
int pthread_barrier_wait(pthread_barrier_t *barrier);
int pthread_barrier_destroy(pthread_barrier_t *barrier);
```

#### $\verb|pthread_barrier_init|$

```c
int pthread_barrier_init(pthread_barrier_t * restrict barrier, const pthread_barrierattr_t * restrict attr, unsigned int count);
```

Crea una nuova barriera con attributi `attr` e per `count` thread (`count` thread partecipano alla barriera). Se `attr=NULL` viene impostato l’attributi di default

#### $\verb|pthread_barrier_wait|$

```c
int pthread_barrier_wait(pthread_barrier_t *barrier);
```

Ogni thread che che invoca questo comando si blocca fino a che `count` thread non ci sono arrivati
Uno di questi riceverà un valore speciale di ritorno: `PTHREAD_BARRIER_SERIAL_THREAD` (gli altri $0$)

#### $\verb|pthread_barrier_destroy|$

```c
int pthread_barrier_destroy(pthread_barrier_t *barrier);
```

Distrugge una barrier, liberando risorse
Ritorna $0$ in caso di esito positivo, altrimenti viene restituito un numero per indicare l’errore

### Condition
La **condition** è un metodo di sincronizzazione che permette ad un thread di sospendere la sua esecuzione finché un predicato su di un dato condiviso non è verificato

```c
int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *cond_attr);
int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *abstime);
int pthread_cond_destroy(pthread_cond_t *cond);
```