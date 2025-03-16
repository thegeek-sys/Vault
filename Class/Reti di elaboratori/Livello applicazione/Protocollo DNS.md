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

### Può essere centralizzato?
No! Per diversi motivi:
- **singolo punto di fallimento** → se crasha il server DNS allora crasha Internet
- **volume di traffico troppo elevato** → un singolo server non potrebbe gestire tutte le query DNS (generate da tutto il mondo)
- **distanza dal database centralizzato** → un singolo server non può essere fisicamente vicino a tutti i client
- **manutenzione** → il server dovrebbe essere aggiornato di continuo per includere i nuovi nomi di host

Dunque un database centralizzato su un singolo server DNS non è *scalabile*

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

### Host aliasing
L'**host aliasing** è un servizio DNS che permette di associare un nome più semplice da ricordare ad un nome più complesso, permettendo ad un host di avere uno o più sinonimi

Ad esempio `relay1.west-coast.enterprise.com` potrebbe avere due sinonimi, quali `enterprise.com` e `www.enterprise.com`. In questo caso `relay1.west-coast.enterprise.com` è un hostname *canonico*, mentre `enterprise.com` e `www.enterprise.com` sono *alias*. Spesso viene fatto per i **mail server**, infatti il mail server e il web server di una società hanno lo stesso alias, ma nomi canonici diversi

Dunque il DNS può essere invocato da un’applicazione per l’hostname canonico di un sinonimo così come l’IP

### Distribuzione del carico
Il DNS viene utilizzato anche per **distribuire il carico** tra server replicati (es. web server). Infatti i siti con molto traffico (es. `cnn.com`) vengono replicati su più server, e ciascuno di questi gira su un sistema terminare diverso e presenta IP differente; per `cnn.com`:
- `151.101.3.5`
- `151.101.67.5`
- `151.101.131.5`
- `151.101.195.5`

Dunque l’hostname canonico (`cnn.com`) è associato ad un insieme di indirizzi IP, e il DNS contiene questo insieme di indirizzi IP. Quando un client effettua una richiesta DNS per un nome mappato ad un insieme di indirizzi, il server risponde con l’insieme di indirizzi ma variando l’ordine ad ogni risposta; questa rotazione permette di distribuire il traffico sui server replicati

---
## Gerarchia server DNS
Nessun server DNS mantiene il mapping per tutti gli host in Internet. Il mapping è infatti distribuito su svariati server DNS

Per garantire un tempo di ricerca veloce si organizzano le informazioni **in base al dominio**
Ci sono 3 classi di server DNS organizzati in una gerarchia:
- **root**
- **top-level domain** (TLD)
- **authoritative** → qui ci sono i mapping veri e propri tra hostname e IP (punto più vicino ai client)

Ci sono poi i server DNS **locali** con cui interagiscono direttamente le applicazioni