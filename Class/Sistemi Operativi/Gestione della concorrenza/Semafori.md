---
Created: 2024-11-29
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
  ---
## Introduction
I semafori sono delle particolari strutture dati su cui si possono fare tre operazioni *atomiche*:
- `initialize`
- `decrement` (o `semWait`) → può mettere il processo in blocked: niente CPU sprecata come il busy waiting
- `increment` (o `semSignal`) → può mettere un processo blocked in ready

Si tratta di syscall, quindi sono eseguite in kernel mode e possono agire direttamente sui processi

### Pseudocodice
```c
struct semaphore {
	int count;
	queueType queue;
};
void semWait(semaphore a) {
	s.count--;
	if (s.count < 0) {
		/* place this process in s.queue */;
		/* block this process */
	}
}
void semSignal(semaphore a) {
	s.count++;
	if (s.count <= 0) {
		/* remove a process P from s.queue */;
		/* place process P on ready list */;
	}
}
```
In questo caso `count` corrisponde al numero di processi che si trovano nella queue

### Semafori binari - pseudocodice
```c
struct binary_semaphore {
	enum {zero, one} value;
	queueType queue;
};
void semWait(binary_semaphore a) {
	if (s.value == one)
		s.value = zero;
	else {
		/* place this process in s.queue */;
		/* block this process */;
	}
}
void semSignalB(binary_semaphore a) {
	if (s.queue is empty())
		s.value = one;
	else {
		/* remove a process P from s.queue */;
		/* place process P on ready list */;
	}
}
```

---
## Semafori - pseudo codice “vero”
![[Pasted image 20241129005046.png]]

Consideriamo tre processi A, B e C
1. A ha già completato la `semWait` e sta eseguendo codice in sezione critica. `s.count` è $0$
2. B entra in `semWait`, count va a $−1$ e B diventa blocked. `s.flag = 0`
3. tocca ad A, che esegue sezione critica e `semSignal`. `s.count = 0`
	- il sistema dunque sposta B che era in wait sul semaforo da blocked a ready
	- A completa `semSignal`. `s.flag = 0`
4. C entra in `semWait`, passa il while `compare_and_swap` e viene interrotto immediatamente dallo scheduler. `s.flag = 1`
5. B riprende l’esecuzione, imposta `s.flag = 0`, esegue la sua sezione critica e chiama `semSignal`. Passa il while. `s.flag = 1` imposta `s.count = 1`, termina `semSignal` ed imposta `s.flag = 0`
Stato corrente: C fermo prima di `s.count−−`; `s.count == 1`; `s.flag == 0`
6. Arriva un nuovo processo D, che entra in `semWait`. Passa il while, ora `s.flag = 1`
7. D esegue `s.count−−`: legge `s.count` da memoria e lo porta in `eax`. `eax == 1` scheduler interrompe D ed esegue C
8. C continua da dov’era: esegue `s.count−−`, che diventa ora $0$, quindi non va in block
9. C preempted nella sezione critica
10. tocca a D, che continua il calcolo di prima: salva `eax − 1` in `s.count`, che rimane $0$
11. D non va in block, e continua nella sezione critica
**Race condition**

---
## Semafori deboli e forti
In base a come scelgo il processo da sbloccare all’interno della coda dei processi si parla di **semafori deboli** e **semafori forti**
I semafori forti sono quelli che usano la politica FIFO (è una coda, manda in esecuzione il processo che aspetta da più tempo)
Ci stanno però sistemi operativi che usano i semafori cosiddetti “deboli” per cui una politica non è specificata, uno qualsiasi dei processi in coda viene sbloccato ma non è dato sapere quale

### Semafori forti - esempio
![[Pasted image 20241129010025.png|350]]
![[Pasted image 20241129010108.png|350]]

---
## Mutua esclusione con i semafori
```c
/* program mutualexclusion */
const int n = /* number of processes */;
semaphore s = 1;

void P(int i) {
	while (true) {
		semWait(s);
		/* critical section */
		semSignal(s);
		/* remainder */
	}
}

void main() {
	parbegin(P(1), P(2), ..., P(n));
}
```
In questo caso non si ha starvation (a meno che i semafori non siano deboli)

![[Pasted image 20241202212139.png]]

