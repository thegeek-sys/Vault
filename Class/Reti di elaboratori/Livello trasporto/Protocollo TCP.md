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
