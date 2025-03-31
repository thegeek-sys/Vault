---
Created: 2025-03-31
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
---
## Introduction
Il protocollo TCP è caratterizzato da:
- pipeline
- bidirezionale (ack in piggybacking)
- orientato al flusso di dati (*stream-oriented*)
- orientato alla connessione
- affidabile (controllo degli errori)
- controllo del flusso
- controllo della congestione

---
## Segmenti TCP
Il TCP riceve i dati da trasmettere sotto forma di byte dal processo (livello applicazione) mittente. Il TCP utilizza il servizio di comunicazione tra host del livello di rete che invia pacchetti; il TCP deve quindi raggruppare un certo numero di byte in **segmenti**, aggiungere un’intestazione e consegnare al livello di rete per la trasmissione

![[Pasted image 20250331231324.png|600]]

### Struttura dei segmenti
![[Pasted image 20250331231427.png]]

>[!info]- Dettaglio
>![[Pasted image 20250331231511.png]]

#### Flag di controllo
I flag di controllo assumono significato quando sono impostati a $1$
![[Pasted image 20250331232019.png|center|500]]

- **URG** → puntatore urgente valido (il puntatore urgente punta ad una sezione di memoria all’interno della zona dati)
- **ACK** → riscontro valido
- **PSH** → richiesta di push (passa i dati al livello applicazione non appena sono ricevuti dal destinatario)
- **RST** → azzeramento della connessione (riguarda il tipo di dato)
- **SYN** → sincronizzazione dei numeri di sequenza (riguarda il tipo di dato)
- **FIN** → chiusura della connessione (riguarda il tipo di dato)

---
## Connessione TCP
La connessione TCP è il percorso virtuale tra il mittente e il destinatario, sopra IP che è privo di connessione

E’ strutturata in 3 fasi:
1. apertura della connessione
2. trasferimento dei dati
3. chiusura della connessione

### Apertura della connessione - 3 way handshake
![[Pasted image 20250331233933.png]]
Il numero di sequenza viene assegnato randomicamente (risulterebbe improbabile che due connessioni abbiamo lo stesso numero di sequenza, in ogni caso cambierebbe il socket)

### Trasferimento dati
#### Push
![[Pasted image 20250331234135.png]]

#### Urgent
I dati **URG** vengono elaborati subito indipendentemente dalla loro posizione nel flusso. Quando il flag è attivo infatti si controlla il campo puntatore urgente (16 bit) che contiene un indirizzo relativo ad una posizione nel campo dati.

Infatti i dati urgenti vengono inseriti all’inizio di un nuovo segmento (che può contenere dati non urgenti a seguire) e il puntatore nell’intestazione indica dove finiscono i dati urgenti ed iniziano quelli normali

>[!example] Interruzione data transfer

### Chiusura della connessione
Ciascuna delle due parti coinvolta nello scambio dati può richiedere la chiusura della connessione (sebbene sia solitamente richiesta dal client), oppure timer nel server (se non si ricevono richieste entro un determinato tempo si chiude la connessione)

La chiusura avviene attraverso un doppio scambio di messaggi FIN e ACK
![[Pasted image 20250331234714.png]]

#### Half close
Si utilizza un half close quando non si è ancora finito di trasferire dati. Infatti il server al posto di inviare FIN e ACK in risposta a FIN del client, invia solo ACK e solo una volta finito il trasferimento verrà inviato FIN

![[Pasted image 20250331235213.png]]

---
## Gestione degli errori
### Numeri di sequenza e ACK
I **numeri di sequenza** indicano il “numero” del primo byte del segmento nel flusso di byte, mentre l’**ACK** indica il numero di sequenza del prossimo byte  atteso dall’altro alto (è un ACK cumulativo)

>[!question] Come gestisce il destinatario i segmenti fuori sequenza?
>La specifica TCP non lo dice, solitamente il destinatario mantiene i byte non ordinati

>[!example] Una semplice applicazione Telnet
>![[Pasted image 20250401003521.png|370]]

### Affidabilità TCP (controllo degli errori)
Per il controllo degli errori nel TCP vengono utilizzati tre mezzi:
- Checksum → se un segmento arriva corrotto viene scartato dal destinatario
- Riscontri e timer di ritrasmissione (RTO) → ack cumulativi e timer associato al più vecchio pacchetto non riscontrato
- Ritrasmissione → ritrasmissione del segmento all’inizio della coda di spedizione

### Generazione di ACK

| Evento                                                                                                                                     | Azione                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. ACK trasmesso in piggy backing                                                                                                          |                                                                                                                                                     |
| 2. Arrivo ordinato di un segmento con numero di sequenza atteso. Tutti i dati fino al numero di sequenza atteso sono già stati riscontrati | ACK delayed. Attende fino a 500 ms l’arrivo del prossimo segmento (per poter riscontrare un ack cumulativo). Se il segmento non arriva, inva un ACK |
| 3. Arrivo ordinato di un segmento con numero di sequenza atteso. Un altro segmento è in attesa di trasmissione dell’ACK (vedi precedente)  | Invia immediatamente un singolo ACK cumulativo, riscontrando entrambi i segmenti ordinati                                                           |
| 4. Arrivo non ordinato di un segmento con numero di sequenza superiore a quello atteso. Viene rilevato un buco                             | Invia immediatamente un ACK duplicato, indicando il numero di sequenza del prossimo byte atteso (ritrasmissione rapida)                             |
| 5. Arrivo di un segmento mancante (uno o più dei successivi è stato ricevuto)                                                              | Invia immediatamente un ACK                                                                                                                         |
| 6. Arrivo di un segmento duplicato                                                                                                         | Invia immediatamente un riscontro con numero di sequenza atteso                                                                                     |
#### Ritrasmissione dei segmenti
Quando un segmento viene inviato, una copia viene memorizzata in una coda di attesa di essere riscontrato (finestra di invio). Se il segmento non viene riscontrato può accadere che:
- scade il timer (è il primo segmento all’inizio della coda) → il segmento viene ritrasmesso e viene riavviato il timer
- vengono ricevuti 3 ack duplicati → ritrasmissione veloce del segmento (senza attendere il timeout)

### FSM mittente
![[Pasted image 20250401004742.png]]

### FSM destinatario
![[Pasted image 20250401004919.png]]

### Normale operatività
![[Pasted image 20250401005143.png]]