---
Created: 2025-04-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
---
## Introduction
Uno sguardo al livello di rete Internet
![[Pasted image 20250423101735.png|center|550]]

Il campo dati dei datagrammi IP può contenere un messaggio ICMP

>[!question] Cosa accade se un router deve scartare un datagramma perché non riesce a trovare un percorso per la destinazione finale?

>[!question] Cosa accade se un datagramma ha il campo TTL  pari a $0$?

>[!question] E se un host di destinazione non ha ricevuto tutti i frammenti di un datagramma entro un determinato limite di tempo?

Come soluzione a tutti questi problemi interviene l’**ICMP** (*Internet Control Message Protocol*). Viene infatti usato a host e router per scambiarsi informazioni a livello di rete