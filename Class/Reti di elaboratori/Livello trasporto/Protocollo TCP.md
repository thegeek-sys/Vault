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

### Segmento smarrito
![[Pasted image 20250401005305.png|600]]

### Ritrasmissione rapida
![[Pasted image 20250401005441.png|600]]

### Riscontro smarrito senza ritrasmissione
![[Pasted image 20250401005628.png|600]]

### Riscontro smarrito con ritramissione
![[Pasted image 20250401005715.png|600]]

### Riassunto sui meccanismi adottati da TCP
- **pipeline** → approccio ibrido tra GBN e ripetizione selettiva
- **numero di sequenza** → primo byte del segmento
- **ACK cumulativo** (conferma tutti i byte precedenti a quello indicato) e **delayed** (posticipato, nel caso di arrivo di un pacchetto in sequenza, con precedenti già riscontrati)
- **timeout** basato su RTT → unico timer di ritrasmissione (associato al più vecchio segmento non riscontrato). Quando arriva un notifica immediata, si riavvia il timer sul più vecchio segmento non riscontrato
- **ritrasmissione**
	- **singola** → solo il segmento non riscontrato (non i successivi)
	- **rapida** → al terzo ACK duplicato prima del timeout si ritrasmette

---
## Controllo del flusso
L’obbiettivo del mittente (per quanto riguarda il controllo del flusso) è quello di non sovraccaricare il buffer del destinatario ritrasmettendo troppi dati, troppo velocemente (bilanciare velocità di invio con velocità di ricezione a livello di processi)

![[Pasted image 20250401011345.png|center|650]]

Per farlo il destinatario invia un feedback esplicito in cui comunica al mittente lo spazio disponibile includendo il valore di **receive window** (*RWND*) nei segmenti (header TCP)

![[Pasted image 20250401011426.png|center|650]]

### Finestra di invio
L’apertura, la chiusura e la riduzione della finestra di invio sono controllare dal destinatario

![[Pasted image 20250401011547.png]]
![[Pasted image 20250401011621.png]]

>[!example] Esempio
>>[!info]
>>Si ipotizza una comunicazione unidirezionale dal client al server. Per questo motivo viene mostrata una sola finestra per lato
>
>![[Pasted image 20250401011737.png]]

---
## Controllo della congestione
Per congestione si intende informalmente che “troppe sorgenti trasmettono troppi dati, a una velocità talmente elevata che la rete non è in grado di gestirli”

Questo problema influenza:
- pacchetti smarriti → overflow nei buffer dei router
- lunghi ritardi → accodamento nei buffer dei router

>[!hint] Tra i dieci problemi più importanti del networking

>[!info] Controllo della congestione vs. controllo del flusso
>Con il controllo del flusso la dimensione della finestra di invio è controllata dal destinatario tramite il valore `rwnd` che viene indicato in ogni segmento trasmesso nella direzione opposta, in modo tale che la finestra del ricevente non viene mai sovraccaricata con i dati ricevuti
>
>I buffer intermedi (nei router) però possono comunque congestionarsi poiché un router riceve dati da più mittenti (non vi è congestione agli estremi ma vi può essere congestione nei nodi intermedi)
>
>La perdita di segmenti comporta la loro rispedizione, aumentando la congestione.
>Dunque la congestione è un problema che riguarda IP ma viene gestito dal TCP

### Approcci
Esistono due principali approcci al controllo della congestione.

Il **controllo di congestione end-to-end** non ha nessun supporto esplicito dalla rete, dunque la congestione è dedotta osservando le perdite e i ritardi nei sistemi terminali (metodo adottato da TCP).
Mentre nel **controllo di congestione assistito dalla rete** i router forniscono feedback ai sistemi terminali, in particolare utilizzano un singolo bit per indicare la congestione (TCP/IP ENC) comunicando in modo esplicito al mittente la frequenza trasmissiva

### Problematiche
1. Come può il mittente limitare la frequenza di invio del traffico sulla propria connessione? [[#1. Finestra di congestione|Finestra di congestione]]
2. Come può il mittente rilevare la congestione? [[#2. Rilevare la congestione|Rilevare la congestione]]
3. Quale algoritmo dovrebbe essere usato per limitare la frequenza di invio in funzione della congestione end-to-end? [[#3. Controllo della congestione|Controllo della congestione]]

### 1. Finestra di congestione
Per controllare la congestione si usa la variabile `CWND` (*congestion window*, relativa alla congestione della rete) che insieme a `RWND` (relativa alla congestione del ricevente) definisce la dimensione della finestra di invio

$$
\text{Dimensione della finestra}=\text{min(rwnd, cwnd)}
$$

### 2. Rilevare la congestione
Per rilevare la congestione si utilizzano, nell’approccio end-to-end, **ACK duplicati** e **timeout** poiché possono essere intesi come eventi di perdita (danno indicazione dello stato della rete)

In particolare se gli **ACK arrivano in sequenza e con buona frequenza**, vuol dire che si può inviare e **incrementare** la quantità di segmenti inviati, se invece si hanno **ACK duplicati o timeout**, vuol dire che è necessario **ridurre** la finestra dei pacchetti che si spediscono senza aver ricevuto riscontri

Dunque si può dire che il TCP è **auto-temporizzante** in quanto reagisce in base ai riscontri che ottiene

### 3. Controllo della congestione
L’idea alla base del controllo della congestione è quella di incrementare il rate di trasmissione se non c’è congestione (ack), diminuire se c’è congestione (segmenti persi)

L’algoritmo di controllo della congestione si basa su tre componenti:
- **slow start**
- **congestion avoidance**
- **fast recovery**

#### Slow start
Nello **slow start** (*incremento esponenziale*) la `CWND` è inizializzata a $1\text{ MSS}$ (maximum segment size) e viene incrementata di $1\text{ MSS}$ per ogni segmento riscontrato

![[Pasted image 20250404103757.png|550]]

>[!example]
>Se arriva un riscontro, $cwnd=cwnd+1$ quindi:
>- Inizio → $cwnd=1\to 2^0$
>- Dopo 1 RTT → $cwnd=cwnd+1=1+1=2\to 2^1$
>- Dopo 2 RTT → $cwnd=cwnd+2=2+2=4\to 2^2$
>- Dopo 3 RTT → $cwnd=cwnd+4=4+4=8\to 2^3$

Ma **fino a quando cresce la CWND**?
La dimensione della finestra di congestione nell’algoritmo slow start viene aumentata esponenzialmente fino al raggiungimento di una soglia (*ssthreshold*)

#### Congestion avoidance
La `CWND` cresce finché non viene perso un pacchetto (in tal caso si pone $\text{ssthreshold}=cwnd/2$, *slow start treshold*). Solo a questo punto si arresta slow start e inizia **congestion avoidance** (*additive increase*)

Con la congestion avoidance si ha un incremento lineare ogni qual volta che viene riscontrata l’intera finestra di segmenti (si incrementa di 1 la `cwnd`) finché non si rileva una congestione (timeout o 3 ack duplicati). Solo a questo punto si imposta $\text{ssthreshold}=\frac{cwnd}{2}$ e $cwnd=1$

![[Pasted image 20250404104906.png|550]]

>[!example]
>Se arriva un riscontro, $cwnd=cwnd+\frac{1}{cwnd}$ quindi:
>- Inizio → $cwnd=i$
>- Dopo 1 RTT → $cwnd=i+1$
>- Dopo 2 RTT → $cwnd=i+2$
>- Dopo 3 RTT → $cwnd=i+3$
>
>Con l’algoritmo congestion avoidance, la dimensione della finestra di congestione viene aumentata linearmente fino alla rilevazione della congestione

### Implementazione TCP - TCP Tahoe
La **TCP Tahoe** considera timeout e 3 ack duplicati come congestione e riparte da $1$ con $\text{ssthreshold}=\frac{cwnd}{2}$

![[Pasted image 20250404105849.png|550]]

#### Affinamento
Per ottimizzare questo processo si potrebbero utilizzare due approcci differenti nel caso in cui la congestione sia lieve o meno.
Infatti 3 ack duplicati indicano la capacità della rete di consegnare qualche segmento (3 pacchetti oltre quelli di cui arriva l’ack sono arrivati), mentre un timeout prima di 3 ack duplicati è “più allarmante” poiché vuol dire che non 