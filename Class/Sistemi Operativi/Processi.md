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

---
