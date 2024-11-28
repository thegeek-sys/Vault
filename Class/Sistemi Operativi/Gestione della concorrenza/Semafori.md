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

