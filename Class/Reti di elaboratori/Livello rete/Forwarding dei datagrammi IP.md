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
	- [[#Introduction#Inviare il datagramma al prossimo hop|Inviare il datagramma al prossimo hop]]
- [[#Aggregazione degli indirizzi|Aggregazione degli indirizzi]]
---
## Introduction
Inoltrare significa collocare il datagramma sul giusto percorso (porta di uscita del router) che lo porterà a destinazione (o lo farà avanzare verso la destinazione)

### Inviare il datagramma al prossimo hop
Si parla di “inviare il datagramma al prossimo hop“ quando un host ha un datagramma da inviare e lo invia al router della rete locale oppure quando un router riceve un datagramma da inoltrare, accedere alla tabella di routing per trovare il successivo hop a cui inviarlo

L’inoltro richiede una riga della tabella per ogni blocco di rete

> [!example]
> ![[Pasted image 20250423095841.png|500]]
> Tabella di inoltro per il router $R1$
> 
> | Indirizzo di rete  | Hop successivo  | Interfaccia |
> | ------------------ | --------------- | ----------- |
> | $180.70.65.192/26$ | -               | m2          |
> | $180.70.65.128/25$ | -               | m0          |
> | $201.4.22.0/24$    | -               | m3          |
> | $201.4.16.0/22$    | -               | m1          |
> | default            | $180.70.65.200$ | m2          |
>
>>[!info] Altra rappresentazione della tabella di inoltro
>>![[Pasted image 20250423100206.png]]
>>
>>All’interno della tabella di routing sono presenti indirizzi di rete (lunghezza inferiore a $32\text{ bit}$), ma un datagramma contiene l’indirizzo IP dell’host di destinazione (pari a $32\text{ bit}$) e non indica la lunghezza del prefisso di rete
>>
>>>[!question] Come si esegue l’instradamento?
>>>Quando arriva un datagramma in cui i $26\text{ bit}$ a sinistra dell’indirizzo di destinazione combaciano con i bit della prima riga, il pacchetto viene inviato attraverso l’interfaccia m2. Analogamente negli altri casi
>>>
>>>La tabella mostra chiaramente che la prima riga ha un prefisso più lungo (che matcha con il successivo) che indica uno spazio di indirizzi più piccolo

---
## Aggregazione degli indirizzi
Inserire nella tabella una riga per ogni blocco può portare a tabelle molto lunghe, con aumento del tempo necessario per effettuare la ricerca. Come soluzione si ha l’**aggregazione degli indirizzi**

> [!example] Aggregazione di indirizzi nella tabella di R2
> 
> ![[Pasted image 20250423101336.png|450]]
> Tabella d’inoltro per R1
> 
> | Indirizzo di rete | Hop successivo  | Interfaccia |
> | ----------------- | --------------- | ----------- |
> | $140.24.7.0/26$   | -               | m0          |
> | $140.24.7.64/26$  | -               | m1          |
> | $140.24.7.128/26$ | -               | m2          |
> | $140.24.7.192/26$ | -               | m3          |
> | $0.0.0.0/0$       | indirizzo di R2 | m4          |
> 
> Tabella d’inoltro per R2
> 
> | Indirizzo di rete | Hop successivo    | Interfaccia |
> | ----------------- | ----------------- | ----------- |
> | $140.24.7.0/24$   | -                 | m0          |
> | $0.0.0.0/0$       | Router di default | m1          |


