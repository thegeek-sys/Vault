---
Created: 2025-03-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
---
## Diagramma di flusso a confronto
![[Pasted image 20250323231944.png]]

---
## Servizi del TCP
I servizi offerti dal TCP sono:
- Comunicazione tra processi (indirizzamento tramite numero di porta)
- Incapsulamento/decapsulamento
- Multiplexing/demultiplexing
- **Trasporto orientato alla connessione**
- **Controllo di flusso**
- **Controllo degli errori**
- **Controllo della congestione**

### Demultiplexing orientato alla connessione
La socket TCP è identificata da 4 parametri:
- indirizzo IP di origine
- numero di porta di origine
- indirizzo IP di destinazione
- numero di porta di destinazione

L’host ricevente usa i quattro parametri per inviare il segmento alla socket appropriata
Un host server può supportare più socket TCP contemporanee (ogni socket è identificata dai suoi 4 parametri)

I server web hanno socket differenti per ogni connessione client (con HTTP non-persistente si avrà una connessione differenza anche per ogni richiesta dallo stesso client)

![[Pasted image 20250323232811.png]]

>[!warning] Porta ≠ socket
>Sulla stessa porta possono essere attive più socket (nell’immagine sopra si hanno 3 socket sulla porta 80)

---
## Servizio connection oriented
Il protocollo TCP garantisce un servizio end-to-end. Viene infatti stabilita una connessione logica prima di scambiarsi i dati

![[Pasted image 20250324001351.png|600]]

---
## Rappresentazione mediante FSM
Il seguente è la rappresentazione di un servizio orientato alla connessione mediante FSM

![[Pasted image 20250324001557.png|600]]

In particolare quello rappresentato nell’immagine è un **three-way handshake**

---
## Controllo di flusso
Quando un’entità produce dati che un’altra entità deve consumare, deve esistere un **equilibrio** fra le velocità di produzione e la velocità di consumo dei dati

Se la velocità di produzione è maggiore della velocità di consumo, il consumatore potrebbe essere sovraccaricato e costretto ad eliminare alcuni dati

Se la velocità di produzione è minore della velocità di consumo, il consumatore rimane in attesa riducendo l’efficienza del sistema

>[!info]
>Il controllo del flusso è legato alla prima problematica per evitare di perdere dati

### Livello di trasporto
A controllare il flusso a livello di trasporto si hanno 4 entità:
- processo mittente
- trasporto mittente
- trasporto desinatario
- processo destinatario

![[Pasted image 20250324002055.png]]

Una soluzione per realizzare un controllo di flusso sta nel **buffer** (insieme di locazioni di memoria che possono contenere pacchetti).
La comunicazione delle informazioni di controllo di flusso può avvenire inviando segnali dal consumatore al produttore

Il livello trasporto del destinatario segnala al livello trasporto del mittente di sospendere l’invio di messaggi quando ha il buffer saturo (quando si libera spazio nel buffer segnala al livello trasporto mittente può riprendere l’invio di messaggi)
Il livello trasporto del mittente segnala al livello applicazione di sospendere l’invio di messaggi quando ha il buffer saturo (quando si libera spazio nel buffer segnala al livello applicazione che può riprendere l’invio di messaggi)

---
## Controllo degli errori
Poiché il livello di rete è inaffidabile, è necessario implementare l’**affidabilità al livello di trasporto**. Per avere un servizio di trasporto affidabile è necessario implementare un **controllo degli errori** (sui pacchetti non bit):
- rilevare e scartare pacchetti corrotti
- tenere traccia dei pacchetti persi e gestirne il rinvio
- riconoscere pacchetti duplicati e scartarli
- bufferizzare i pacchetti fuori sequenza finché arrivano i pacchetti mancanti

Il controllo degli errori coinvolge solo il livello trasporto mittente e destinatario (i messaggi scambiati tra livelli sono esenti da errori)
Il livello trasporto del destinatario gestisce il controllo degli errori segnalando il problema al livello trasporto del mittente

![[Pasted image 20250324002827.png|500]]

### Realizzazione
Il mittente deve sapere quali pacchetti ritrasmettere e il destinatario deve saper riconoscere pacchetti duplicati e fuori sequenza
Per fare ciò vengon numerai i pacchetti con il **numero di sequenza** (campo all’interno dell’header) che rispetta una numerazione sequenziale

Poiché il numero di sequenza deve essere inserito nell’intestazione del pacchetto, occorre specificare la dimensione massima. Se l’intestazione prevede $m$ bit per il numero di sequenza questi possono assumere i valori da $0$ a $2^m-1$. I numeri di sequenza sono considerati in modulo $2^m$ (una volta raggiunto il numero massimo ricomincia la numerazione da $0$, numerazione circolare)

Il numero di sequenza è utile al destinatario per capire:
- la sequenza di pacchetti in arrivo
- i pacchetti persi
- i pacchetti duplicati

Per capire se un pacchetto è andato perso, il mittente usa il **numero di riscontro** (*acknowledgement*, *ack*) che permette di notificare al mittente la corretta ricezione di un pacchetto
Il destinatario può scartare i pacchetti corrotti e duplicati

---
## Integrazione del controllo degli errori e controllo di flusso
Il controllo di flusso richiede due buffer (mittente e destinatario) mentre il controllo degli errori richiede il numero di sequenza e ack
La combinazione dei due meccanismi crea un **buffer numerato** (presso mittente e destinatario)

**Mittente**:
- quando prepara un pacchetto usa come numero di sequenza il numero ($x$) della prima locazione libera del buffer
- quando invia il pacchetto ne memorizza una coppia nella locazione $x$
- quando riceve un ack di un pacchetto libera la posizione di memoria che era occupata da quel pacchetto

**Destinatario**:
- quando riceve un pacchetto con numero di sequenza $y$, lo memorizza nella locazione $y$ fin quando il livello applicazione è pronto a riceverlo
- quando passa il pacchetto $y$ al livello applicazione invia ack al mittente

Poiché i numeri di sequenza sono calcolati in modulo $2^m$, possono essere rappresentati con un cerchio. Il buffer viene rappresentato con un insieme di settori, chiamati *sliding windows*, che in ogni istante occupano una parte del cerchio

