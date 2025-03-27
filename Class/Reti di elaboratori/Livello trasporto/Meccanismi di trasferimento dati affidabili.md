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

