---
Created: 2025-04-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Nel dettaglio|Nel dettaglio]]
- [[#Traceroute e ICMP|Traceroute e ICMP]]
	- [[#Traceroute e ICMP#Criteri di arresto dell’invio|Criteri di arresto dell’invio]]
---
## Introduction
Uno sguardo al livello di rete Internet
![[Pasted image 20250423101735.png|center|550]]

Il campo dati dei datagrammi IP può contenere un messaggio ICMP

>[!question] Cosa accade se un router deve scartare un datagramma perché non riesce a trovare un percorso per la destinazione finale?

>[!question] Cosa accade se un datagramma ha il campo TTL  pari a $0$?

>[!question] E se un host di destinazione non ha ricevuto tutti i frammenti di un datagramma entro un determinato limite di tempo?

Come soluzione a tutti questi problemi interviene l’**ICMP** (*Internet Control Message Protocol*). Viene infatti usato a host e router per scambiarsi informazioni a livello di rete (ora il router può anche generare pacchetti, prima li poteva solo inoltrare)

>[!example]
>![[Pasted image 20250423102328.png]]
>
>Un uso tipico dell'ICMP è fornire un meccanismo di feedback quando viene inviato un messaggio IP. In questo esempio, il dispositivo $A$ sta cercando di inviare un datagramma IP al dispositivo $B$. Tuttavia, quando arriva al router $R3$, viene rilevato un problema di qualche tipo che causa l'eliminazione del datagramma.
>**$R3$ invia un messaggio ICMP indietro ad $A$ per informarlo che è successo qualcosa,** auspicabilmente con informazioni sufficienti per permettere ad A di correggere il problema, se possibile.
>
>**$R3$ può inviare il messaggio ICMP solo ad $A$, non a $R2$ o $R1$.**

---
## Nel dettaglio
Dunque l’ICMP viene usato da host e router per scambiarsi informazioni a livello di rete (report degli errori, echo request/reply). Inoltre **ICMP è considerato parte di IP anche se usa IP per inviare i suoi messaggi**

I messaggi ICMP hanno un campo tipo e un campo codice, e contengono l’intestazione e i primi 8 byte del datagramma IP che ha provocato la generazione del messaggio.

![[Pasted image 20250423102822.png|350]]

---
## Traceroute e ICMP
Il programma **traceroute** invia una serie di datagrammi IP alla destinazione ciascuno contenente un segmento UDP con un numero di porta inutilizzata (il primo pari a $TTL=1$, il secondo pari a $TTL=2$, ecc.) e avvia un timer per ogni datagramma

Quando l’$n$-esimo datagramma arriva all’$n$-esimo router:
- il router scarta il datagramma.
- invia all’origine un messaggio di allerta ICMP (tipo 11, codice 0) che include il nome del router e l’indirizzo IP
- quando il messaggio ICMP arriva, l’origine può calcolare RTT

Tutto questo viene eseguito dal traceroute per 3 volte

### Criteri di arresto dell’invio
L’invio viene arrestato quando l’host di destinazione restituisce un messaggio ICMP di porta non raggiungibile (tipo 3, codice 3)

>[!example]
>![[Pasted image 20250423103503.png]]

