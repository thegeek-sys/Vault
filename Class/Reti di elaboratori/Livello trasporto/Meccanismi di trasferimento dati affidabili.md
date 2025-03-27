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
![[immagine_sfondo_bianco.png]]

### Efficienza
Consideriamo il prodotto $\text{rate}\cdot \text{ritardo}$ (misura del numero di bit che il mittente può inviare prima di ricevere un ack, volume della pipe in bit). Se il rate è elevato e il ritardo consistente allora lo stop and wait sarà inefficiente

>[!example]
>In un sistema che utilizza stop and wait abbiamo:
>- rate → $1 \text{ Mbps}$
>- ritardo di andata e ritorno di $1\text{ bit}$ → $20\text{ ms}$
>
>Quanto vale $\text{rate}\cdot \text{ritardo}$?
>Se i pacchetti hanno dimensione $1000\text{ bit}$, qual è la percentuale di utilizzo del canale
>
>$\text{rate}\cdot \text{ritardo}=(1\times 10^6)\times(20\times 10^{-3})=20000\text{ bit}$
>Il mittente potrebbe inviare $200000\text{ bit}$ nel tempo necessario per andare dal mittente al ricevente e viceversa ma ne invia solo $1000$
>
>Il **coefficiente di utilizzo** del canale è $\frac{1000}{200000}=5\%$, risultando molto inefficiente

---
## Protocolli con pipeline