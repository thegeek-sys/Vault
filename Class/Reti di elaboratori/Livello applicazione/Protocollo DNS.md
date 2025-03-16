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
Note le premesse, si è deciso di memorizzare i dati in un **database distribuito** implementato in una gerarchia di server DNS che è accessibile tramite una **protocollo a livello applicazione** che consente agli host di interrogare il database distribuito per *risolvere i nomi* (tradurre indirizzi/nomi)

Il DNS viene utilizzato dagli altri protocolli di livello applicazione (HTTP, SMTP, FTP) per tradurre gli hostname in indirizzi IP. Utilizza il trasposto UDP (lo vedremo nel livello di trasporto) e indirizza la porta 53

>[!example] Esempio di interazione con HTTP
>Un browser (ossia client HTTP) di un host utente richiede la URL `www.someschool.edu`
>1. L’host esegue il lato client dell’applicazione DNS
>2. Il browser estrae il nome dell’host `www.someschool.edu` dall’URL e lo passa al lato client dell’applicazione DNS
>3. Il client DNS invia una query contenente l’hostname a un server DNS
>4. Il client DNS riceve una risposta che include l’indirizzo IP corrispondente all’hostname
>5. Ottenuto l’indirizzo IP dal DNS, il browser può dare inizio alla connessione TCP verso il server HTTP localizzato a quell’indirizzo IP