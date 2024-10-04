---
Created: 2024-10-03
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Requisiti di un SO
Il compito fondamentale di un sistema operativo è quello di **gestire i processi**, deve quindi gestire tutte le varie computazioni.
Un sistema operativo moderno deve quindi:
- permettere l’esecuzione alternata di processi multipli → anche se ho meno processori che processi questo non deve essere un problema, bisogna far alternare i processi in esecuzione sui processori disponibili
- assegnare le risorse ai processi (es. un processo richiede l’uso della stampante)
- permettere ai processi di scambiarsi informazioni
- permettere la sincronizzazione tra processi

---
## Cos’è un processo?
Un **processo** è un’istanza di un programma, memorizzato generalmente sul archiviazione di massa, in esecuzione (ogni singola istanza di fatti dà vita a un processo).
Potrebbe però anche essere visto come un’entità che può essere assegnata ad un processore in esecuzione tipicamente caratterizzata dall’esecuzione da una sequenza di istruzioni di cui voglio conoscere l’esito, uno stato corrente e da un insieme associato di risorse

Questo è quindi composto da:
- **codice** → ovvero le istruzioni da eseguire
- un insieme di **dati**
- un numero di **attributi** che descrivono il suo stato

Un processo ha 3 macrofasi: creazione, esecuzione, terminazione. Quest’ultima può essere prevista (quando il programma è terminato o quando l’utente chiude volontariamente in programma tramite la X) oppure non prevista (processo esegue un’operazione non consentita che potrebbe risultare con una terminazione involontaria del processo)

---
## Elementi di un processo
Finché un processo è in esecuzione ad esso sono associati un certo insieme di informazioni, tra cui:
- identificatore
- stato (running etc.)
- priorità
- hardware context → attuale situazione dei registri
- puntatori alla memoria principale (definisce l’immagine del processo)
- informazioni sullo stato dell’input/output
- informazioni di accounting (quale utente ha eseguito il processo)

### Process Control Block
Per ciascuno processo attualmente in esecuzione è presente un **process control block**. Si tratta di un insieme di informazioni (gli elementi di un processo) raccolte insieme e mantenute nella zona di memoria riservata al kernel.
Questo viene interamente creato e gestito dal sistema operativo e il suo scopo principale è quello di permettere al SO di **gestire più processi contemporaneamente** (contiene  infatti le  informazioni sufficienti per bloccare un programma e farlo riprendere più tardi dallo stesso punto in cui si trovava)

---
## Traccia di un processo
Un ulteriore aspetto importante in un processo è la **trace** ovvero l’insieme di istruzioni di cui è costituito un processo. Il **dispatcher** invece è un piccolo programma che sospende un processo per farne andare un altro in esecuzione

### Esecuzione di un processo
Si considerino 3 processi in esecuzioni, tutti caricati in memoria principale
![[Screenshot 2024-10-04 alle 10.57.44.png|140]]

La traccia, dal punto di vista del processo, appare come l’esecuzione sequenziale delle istruzioni del singolo processo
![[Screenshot 2024-10-04 alle 11.00.20.png]]

La traccia, del punto di visto del processore, ci mostra come effettivamente vengono eseguiti i 3 processi
![[Screenshot 2024-10-04 alle 11.02.08.png]]
>[!note] Le righe in blu sono gli indirizzi del dispatcher

---
## Modello dei processi a 2 stati
Un processo potrebbe essere in uno di questi due stati
- in esecuzione
- non in esecuzione (anche quando viene messo in pausa dal dispatcher)
![[Screenshot 2024-10-04 alle 11.05.44.png|center|450]]

Dal punto di vista dell’implementazione avremmo una coda in cui sono processi tutti i processi che non sono in esecuzione, il dispatch quindi prende il processo in cima alla queue (quello in nero) e lo mette in esecuzione.
I processi vengono quindi mossi dal dispacher dalla CPU alla coda e viceversa, finché il processo non viene completato
![[Screenshot 2024-10-04 alle 11.09.52.png|400]]
![[Screenshot 2024-10-04 alle 11.08.22.png|400]]

---
## Creazione e terminazione di processi
### Creazione
In ogni istante in un sistema operativo sono $n\geq 1$ processi in esecuzione (come minimo ci sta un’interfaccia grafica o testuale in attesa di input). Quando viene dato un comando dall’utente, quasi sempre si crea un nuovo processo

**process spawning** → è un fenomeno che si verifica quando un processo in esecuzione crea un nuovo processo