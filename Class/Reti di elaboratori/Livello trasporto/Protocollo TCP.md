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
