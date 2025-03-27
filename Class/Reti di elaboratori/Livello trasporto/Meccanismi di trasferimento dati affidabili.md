---
Created: 2025-03-27
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Stop-and-wait
Lo **stop-and-wait** è un meccanismo di trasferimento dati orientato alla connessione con controllo di flusso e controllo degli errori

In questo caso mittente e destinatario per comunicare usano una **finestra scorrevole di dimensione $1$**. Il mittente invia un pacchetto alla volta e ne attende l’*ack* prima di spedire il successivo.

Quando il pacchetto arriva al destinatario viene calcolato il checksum. In caso il checksum corrisponda viene inviato l’ack al mittente, ma in caso contrario il pacchetto viene scartato senza informare il mittente. Infatti, per capire se un pacchetto è andato perso il mittente usa un **timer**; una volta che è scaduto il timer senza ricevere ack viene rinviato il pacchetto

![[Pasted image 20250327233134.png|center|650]]

Il mittente deve tenere una copia del pacchetto spedito finché non riceve riscontro

### Numeri di sequenza e riscontro
Per gestire i pacchetti duplicati lo stop&wait utilizza i numeri di sequenza. Per fare ciò si vuole identificare l’intervallo più piccolo possibile che possa consentire la comunicazione senza ambiguità.

Supponiamo che il mittente abbia inviato il pacchetto con numero di sequenza $x$. Si possono verificare 3 casi:
1. Il pacchetto arriva correttamente al destinatario che invia un riscontro. Il riscontro arriva al mittente che invia il pacchetto successivo numerato $x+1$
2. Il pacchetto risulta corrotto o non arriva al destinatario. Il mittente allo scadere del timer invia nuovamente il pacchetto $x$
3. Il pacchetti arriva correttamente al destinatario ma il riscontro viene perso o corrotto. Scade il timer e il mittente rispedisce $x$. *Il destinatario riceve un duplicato, se ne accorge?*

I numeri di sequenza $0$ e $1$ sono sufficienti per il protocollo stop and wait.
Come convenzione infatti si è scelto che il numero di riscontro (ack) indica il numero di sequenza del prossimo pacchetto atteso dal destinatario (pacchetto che deve arrivare).

>[!example]
>Se il destinatario ha ricevuto correttamente il pacchetto $0$ invia un riscontro con valore $1$ (che significa che il prossimo pacchetto atteso ha numero di sequenza $1$)

>[!hint]
>Nel meccanismo stop and wait, il numero di riscontro indica, in aritmetica modulo 2, il numero di sequenza del prossimo pacchetto atteso dal destinatario

### FSM mittente
![[Pasted image 20250327234050.png]]
Una volta inviato un pacchetto, il mittente si blocca (non spedisce pacchetto successivo) e aspetta (stop and wait) finché non riceve ack

### FSM destinatario
![[Pasted image 20250327234149.png]]
Il destinatario è sempre nello stato ready

### Diagramma di comunicazione
