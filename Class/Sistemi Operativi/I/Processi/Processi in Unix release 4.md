---
Created: 2024-11-26
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Class/Sistemi Operativi/I/Processi/Processi]]"
Completed: 
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Transizioni tra stati dei processi|Transizioni tra stati dei processi]]
- [[#Processo Unix|Processo Unix]]
	- [[#Processo Unix#Livello utente|Livello utente]]
	- [[#Processo Unix#Livello registro|Livello registro]]
	- [[#Processo Unix#Livello sistema|Livello sistema]]
- [[#Process Table Entry|Process Table Entry]]
- [[#U-Area|U-Area]]
- [[#Creazione di un processo in Unix|Creazione di un processo in Unix]]
---
## Introduction
Utilizza la seconda opzione in cui la maggior parte del SO viene eseguito all’interno dei processi utente in modalità kernel

---
## Transizioni tra stati dei processi
![[Screenshot 2024-10-08 alle 00.09.44.png]]
Dal kernel running si può passare a preempted, ovvero quel momento prima che finisca il processo in cui il kernel per qualche motivo decide di togliergli il processore.
Quando un processo finisce, prima che muoia, va nello stato zombie, in cui tutta la memoria di quel processo viene deallocata (compresa l’immagine) e l’unica cosa che sopravvive è il process control block con l’unico scopo di comunicare l’exit status al padre; una volta che il padre ha ricevuto che il figlio gli ha dato questo exit status, a quel punto anche il PCB viene tolto e il processo figlio viene definitivamente terminato
Da notare che un processo in kernel mode non è interrompibile che non lo rendeva adatto ai processi real-time

In sintesi
**User running** → in esecuzione in modalità utente; per passare in questo stato bisogna necessariamente passare per kernel running in quanto è avvenuto un process switch, l’unica cosa che può avvenire è tornare in kernel running in seguito ad una system call o interrupt
**Kernel running** → in esecuzione in modalità kernel o sistema
**Ready to Run, in Memory** → può andare in esecuzione non appena il kernel lo seleziona
**Asleep in Memory** → non può essere eseguito finché un qualche evento non si manifesta e ci è diventato a seguito di un evento bloccante; il processo è in memoria, corrisponde al blocked del modello a 7 stati
**Ready to Run, Swapped** → può andare in esecuzione (non è in attesa di eventi esterni), ma prima dovrà essere portato in memoria
**Sleeping, Swapped** → non può essere eseguito finché un qualche evento non si manifesta; il processo non è in memoria primaria
**Preempted** → il kernel ha appena tolto l’uso del processore a questo processo (*preemption*), per fare un context switch
**Created** → appena creato, ma non ancora pronto all’esecuzione
**Zombie** → terminato tutta la memoria del processo viene deallocata (compresa l’immagine) e l’unica cosa che sopravvive è il process control block con l’unico scopo di comunicare l’exit status al padre; una volta che il padre lo ha ricevuto, anche il PCB viene tolto e il processo figlio viene definitivamente terminato

---
## Processo Unix
Un processo in unix è diviso in:
- livello utente
- livello registro
- livello di sistema

### Livello utente
**Process text** → il codice sorgente (in linguaggio macchina) del processo
**Process data** → sezione di dati del processo; compresi anche i valori delle variabili
**User stack** → stack delle chiamate del processo; in fondo contiene anche gli argomenti  con cui il processo è stato invocato
**Shared memory** → memoria condivisa con altri processi, usata per le comunicazioni tra processi

### Livello registro
**Program counter** → indirizzo della prossima istruzione del process text da eseguire
**Process status register** → registro di stato del processore, relativo a  quando è stato swappato l’ultima volta
**Stack pointer** → puntatore alla cima dello user stack
**General purpose registers** → contenuto dei registri accessibili al programmatore, relativo a quando è stato swappato l’ultima volta

### Livello sistema
**Process table entry** → puntatore alla tabella di tutti i processi, dove individua quello corrente
**U area** → informazioni per il controllo del processo
**Per process region table** → definisce il mapping tra indirizzi virtuali ed indirizzi fisici (page table)
**Kernel stack** → stack delle chiamate, separato da quello utente, usato per le funzioni da eseguire in modalità sistema

---
## Process Table Entry
![[Screenshot 2024-10-08 alle 00.39.39.png]]

---
## U-Area
![[Screenshot 2024-10-08 alle 00.40.22.png]]

---
## Creazione di un processo in Unix
La creazione di un processo unix tramite una chiamata di sistema `fork()`. In seguito a ciò, in Kernel Mode:
1. Alloca una entry nella tabella dei processi per il nuovo processo (figlio)
2. Assegna un PID unico al processo figlio
3. Copia l’immagine del padre, escludendo dalla copia la memoria condivisa (se presente)
4. Incrementa i contatori di ogni file aperto dal padre, per tenere conto del fatto che ora sono anche del figlio
5. Assegna al processo figlio lo stato Ready to Run
6. Fa ritornare alla fork il PID del figlio al padre, e 0 al figlio

Quindi, il kernel può scegliere tra:
- continuare ad eseguire il padre
- switchare al figlio
- switchare ad un altro processo

>[!info]
>Creare un processo a partire dal processo padre è il modo più efficiente di avviare un processo in quanto la maggior parte delle volte un programma inizia un processo a partire dal codice sorgente già esistente
