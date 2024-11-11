---
Created: 2024-11-11
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Index
- [[#Tecniche per effettuare l’I/O|Tecniche per effettuare l’I/O]]
	- [[#Tecniche per effettuare l’I/O#Direct Memory Access (DMA)|Direct Memory Access (DMA)]]
- [[#Evoluzione della funzione di I/O|Evoluzione della funzione di I/O]]
---
## Tecniche per effettuare l’I/O
Ci sono sostanzialmente quattro modalità di fare I/O

|                             | **Senza interruzioni** | **Con interruzioni**           |
| --------------------------- | ---------------------- | ------------------------------ |
| **Passando per la CPU**     | I/O programmato        | I/O guidato dalle interruzioni |
| **Direttamente in memoria** |                        | DMA                            |

### Direct Memory Access (DMA)
Quello sotto è un tipico modulo DMA
![[Pasted image 20241108193610.png|250]]

Il processore delega le operazioni di I/O al modulo DMA il quale trasferisce direttamente i dati da o verso la memoria principale. Quando l’operazione è completata il DMA genera un interrupt per il processore

---
## Evoluzione della funzione di I/O
Riassumiamo brevemente quale è stata l’evoluzione dell’I/O in ordine cronologico
1. Il processore controlla il dispositivo periferico
2. Viene aggiunto un modulo (o controllore) di I/O direttamente sul dispositivo → permettendo di fare I/O programmato senza interrupt (no multiprogrammazione), ma il processore non si deve occupare di alcuni dettagli del dispositivo stesso
3. Modulo o controllore di I/O con interrupt → migliora l’efficienza del processore, che non deve aspettare il completamento dell’operazione di I/O
4. DMA → i blocchi di dati viaggiano tra dispositivo e memoria senza usare il processore
5. Il modulo di I/O diventa un processore  separato (*I/O channel*) → il processore “principale” comanda al processore di I/O di eseguire un certo programma di I/O in memoria principale
6. Processore per l’I/O (*I/O processor* o anche *I/O channel*) → ha una sua memoria dedicata ed è usato per le comunicazioni con terminali interattivi

Nell’architettura moderna il chipset implementa le funzioni di interfaccia I/O
![[Pasted image 20241108200410.png]]