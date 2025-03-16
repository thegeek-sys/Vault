---
Created: 2025-03-15
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello applicazione]]"
Completed:
---
---
## Identificazione degli host
Gli host internet hanno nomi (*hostname*) che sono facili da ricordare ma forniscono poca informazione sulla collocazione degli host all’interno di Internet (`w3.uniroma1.it` ci dice che l’host si trova probabilmente in Italia ma non dove)
Per questo motivo esistono gli **indirizzi IP** per gli host, delle sequenze a $32\text{ bit}$, usati per indirizzare i datagrammi

### Indirizzo IP
Un indirizzo IP consiste di $4\text{ byte}$, ed è costituito da una stringa in cui ogni punto separa uno dei byte espressi con un numero decimale compreso tra $0$ e $255$
Questo presenta una **struttura gerarchica** composta da:
- rete di appartenenza (prefisso, numero di bit variabile)
- indirizzo nodo (bit restanti)

---
## DNS: Domain Name System
Il **DNS** (*Domain Name System*) ha il compito di associare un hostname al relativo indirizzo ip

![[Pasted image 20250315174220.png]]

Ai tempi di ARPANET era un file `host.txt` che veniva caricato durante la notte (erano pochi indirizzi), adesso è un’applicazione che gira su ogni host costituita da un gran numero di server DNS distribuiti per il mondo e un protocollo a livello applicazione che specifica la comunicazione tra server DNS e host richiedenti

### E’ un applicazione?
E’ un protocollo del livello applicazione che viene eseguito dagli end systems secondo il paradigma client-server. Utilizza un protocollo end-to-end per trasferire messaggi tra gli end system (UDP).
Non è un’applicazione con cui gli utenti interagiscono direttamente (ecctto gli amministratori di rete), ma fornisce funzionalità di base di internet per le applicazione complesse
Inoltre rispecchia la filosofia di concentrare la complessità nelle parti delle periferiche di rete

---
## Servizio DNS
