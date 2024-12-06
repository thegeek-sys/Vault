---
Created: 2024-12-06
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Introduction
Nel problema dei lettori/scrittori si ha un’**area dati condivisa** tra molti processi di cui **alcuni la leggono, altri la scrivono**

Condizioni da soddisfare:
- più lettori possono leggere l’area contemporaneamente (nei produttori/consumatori non era permesso)
- solo uno scrittore può scrivere nell’area
- se uno scrittore è all’opera sull’area, nessun lettore può effettuare letture

La vera grande differenza con i produttori/consumatori sta nel fatto che l’area condivisa **si accede per intero** (niente problemi di buffer pieno o vuoto, ma è importante permettere ai lettori di accedere contemporaneamente)

---
## Soluzione con precedenza ai lettori
![[Pasted image 20241206232642.png|center|400]]

Il `writer` ha come unico compito quello di scrivere e lo fa attraverso un semaforo di mutua esclusione (solo un `writer` per volta può scirvere)
Il `reader` invece (come per la soluzione di [[Semafori#Trastevere|trastevere]]) incrementa il valore di `readcount` e se si tratta del primo reader, vengono bloccati eventuali scrittori (ma se uno scrittore si trova già nella sezione critica è il lettore a bloccarsi). Viene quindi eseguita l’operazione di lettura e infine viene decrementato il valore di `readcount` e se si tratta dell’ultimo lettore, vengono sbloccati eventuali `writer`

Potrebbe accadere in questa soluzione che si vada in starvation sui `writer`

---
## Soluzione con precedenza agli scrittori
![[Screenshot 2024-12-06 alle 23.35.16.png]]

In questo caso ho fatto ciò che prima avevo fatto solo per i `reader`, per i `writer`
Per i `reader` invece è stato aggiunto, oltre al solito semaforo locale, il semaforo `rsem` che permette, essendo `writer` controllato da un semaforo locale, di evitare che il lettore possa prevalere sullo scrittore; infatti se ad esempio, dopo aver decrementato `readcount` arriva un altro lettore, e dovrebbe continuare a leggere. Immaginiamo però che nel mentre sia arrivato uno scrittore, visto che è arrivato uno scrittore viene fatto `wait(rsem)` che blocca agli ulteriori lettori di entrare nella coda. Viene quindi finita la coda di quelli erano già riusciti a leggere, che quindi sbloccheranno il `writer`

---
## Soluzione con i messaggi
```c
// mailbox = readrequest, writerequest, finished, controller_pid
// send non bloccante, receive bloccante

void reader(int i) {
	while(true) {
		nbsend(readrequest, null);
		receive(controller_pid, null);
		READUNIT();
		nbsend(finished, null);
	}
}

void writer(int j) {
	while(true) {
		nbsend(writerequest, null);
		receive(controller_pid, null);
		WRITEUNIT();
		nbsend(finished, null);
	}
}

void controller() {
	int count = MAX_READERS;
	while(true) {
		if (count > 0) {
			if (!empty(finished)) {
				/* da reader! */
				receive(finished, msg);
				count++;
			}
			else if (!empty(writerequest)) {
				receive(writerequest, msg);
				writer_id = msg.sender;
				count = count - MAX_READERS;
			}
			else if (!empty(readrequest)) {
				receive(readrequest, msg);
				count--;
				nbsend(msg.sender, "OK");
			}
		}
		if (count == 0) {
			nbsend(writer_id, "OK");
			receive(finished, msg); /* da writer! */
			count = MAX_READERS;
		}
		while (count < 0) {
			receive(finished, msg); /* da reader! */
			count++;
		}
	}
}
```