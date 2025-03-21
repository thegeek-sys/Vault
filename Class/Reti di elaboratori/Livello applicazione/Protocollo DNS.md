---
Created: 2025-03-15
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello applicazione]]"
Completed:
---
---
## Index
- [[#Identificazione degli host|Identificazione degli host]]
	- [[#Identificazione degli host#Indirizzo IP|Indirizzo IP]]
- [[#DNS: Domain Name System|DNS: Domain Name System]]
	- [[#DNS: Domain Name System#E’ un applicazione?|E’ un applicazione?]]
	- [[#DNS: Domain Name System#Può essere centralizzato?|Può essere centralizzato?]]
	- [[#DNS: Domain Name System#Perché UDP?|Perché UDP?]]
- [[#Servizio DNS|Servizio DNS]]
	- [[#Servizio DNS#Host aliasing|Host aliasing]]
	- [[#Servizio DNS#Distribuzione del carico|Distribuzione del carico]]
- [[#Gerarchia server DNS|Gerarchia server DNS]]
	- [[#Gerarchia server DNS#Server radice|Server radice]]
	- [[#Gerarchia server DNS#Server TLD e server di competenza|Server TLD e server di competenza]]
	- [[#Gerarchia server DNS#Etichette dei domini generici|Etichette dei domini generici]]
	- [[#Gerarchia server DNS#Server DNS locale|Server DNS locale]]
- [[#Query ricorsiva|Query ricorsiva]]
	- [[#Query ricorsiva#Caching|Caching]]
- [[#DNS record e messaggi|DNS record e messaggi]]
	- [[#DNS record e messaggi#Record DNS|Record DNS]]
		- [[#Record DNS#Type=A|Type=A]]
		- [[#Record DNS#Type=CNAME|Type=CNAME]]
		- [[#Record DNS#Type=NS|Type=NS]]
		- [[#Record DNS#Type=MX|Type=MX]]
		- [[#Record DNS#Tipi di record|Tipi di record]]
	- [[#DNS record e messaggi#Messaggi DNS|Messaggi DNS]]
- [[#Inserire record nel database DNS|Inserire record nel database DNS]]
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

### Perché UDP?
- Less overhead (traffico aggiuntivo non usato per trasmettere dati):
	- Messaggi corti
	- Tempo per set-up connessione di TCP lungo
	- Un unico messaggio deve essere scambiato tra una coppia di server (nella risoluzione contattati diversi server, se si usasse TCP ogni volta dovremmo mettere su la connessione)
- Se un messaggio non ha risposta entro un timeout semplicemente viene ri-inviato dal resolver (problema risolto dallo strato applicativo)

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

![[DNS.png]]

>[!example]
>Il client vuole l’IP di `www.amazon.com`
>1. Il DNS interroga il server root per trovare il server DNS `com`
>2. Il client interroga il server DNS `com` per ottenere il server DNS `amazon.com`
>3. Il client interroga il server DNS `amazon.com` per ottenere l’indirizzo IP di `www.amazon.com`

### Server radice
In Internet ci sono $13$ server DNS radice. Ognuno di questi server è replicato per motivi di sicurezza e affidabilità (in totale diventano 247 root server)
I root server vengono contattati dei server DNS locali, il server DNS radice quindi:
- contatta un server DNS TLD se non conosce la mappatura
- ottiene la mappatura
- restituisce la mappatura al server DNS locale

![[Pasted image 20250316172952.png]]

### Server TLD e server di competenza
I **server TLD** (*top-level domain*) si occupano dei domini `com`, `org`, `net`, ecc. e di tutti i domini locali di alto livello quali `it`, `uk`, `fr`, `ca` e `jp`
La compagnia Verisign Global Registry Services gestisce i server TLD per il dominio `com`, mentre il Registro.it che ha sede a Pisa all’Istituto di Informatica e Telematica (CNR) gestisce il dominio `it`

I **server di competenza** (*authoritative server*) li ha ogni organizzazione dotata di host Internet pubblicamente accessibili (server web e server di posta) e hanno il compito di fornire i record DNS di pubblico dominio che mappano i nomi di tali host in indirizzi IP
Questi possono essete mantenuti dall’organizzazione (università) o da un service provider. In generale sono due server (primario e secondario) per ridondanza

![[Pasted image 20250316173624.png|600]]

### Etichette dei domini generici
![[Pasted image 20250316173719.png]]

### Server DNS locale
Il server **DNS locale** non appartiene strettamente alla gerarchia dei server, ma ciascun ISP (università, società, ISP residenziale) ha un server DNS locale (detto anche “*default name server*“)

Quando un host effettua una richiesta DNS, la query viene inviata al suo server DNS locale, il quale opera da proxy e inoltra la query in una gerarchia di server DNS

---
## Query ricorsiva
In una query ricorsiva si affida il compito di tradurre il nome al server DNS contattato

![[Pasted image 20250316174149.png|center|400]]

### Caching
Il DNS sfrutta il caching per migliorare le prestazioni di ritardo e per ridurre il numero di messaggi DNS che “rimbalzano” in Internet.
Una volta che un server DNS impara la mappatura, la mette nella cache. Le informazioni nella cache vengono invalidate (spariscono) dopo un certo periodo di tempo e tipicamente un server DNS locale memorizza nella cache gli indirizzi IP dei server TLD (ma anche quelli di competenza), quindi i server DNS radice non vengono visitati spesso

I meccanismi di aggiornamento/notifica sono progettati da IETF

---
## DNS record e messaggi
Il mapping è mantenuto nei database sotto forma di **resource record** (RR).
Ogni RR mantiene un mapping (es. tra hostname e indirizzo IP, alias e nome canonico, etc.)

I record vengono spediti tra server e all’host  richiedente all’interno di **messaggi DNS**. Un messaggio può contenere più RR

### Record DNS
Dunque il database distribuito ha il compito di memorizzare i resource record
Ogni messaggio di risposta DNS trasporta uno o più RR

![[Pasted image 20250316174904.png|center|600]]

#### Type=A
$$
\text{Hostname} \Rightarrow \text{IP \textcolor{red}{a}ddress}
$$
All’interno di questo tipo di record:
- `name` → nome dell’host (nome canonico)
- `value` → indirizzo IP

>[!example] `(relay1.bar.foo.com, 45.37.93.126, A)`

#### Type=CNAME
$$
\text{Alias} \Rightarrow \text{\textcolor{red}{C}anonical \textcolor{red}{Name}}
$$
All’interno di questo tipo di record:
- `name` → alias di qualche nome canonico
- `value` → nome canonico

>[!example] `(foo.com, relay1.bar.foo.com, CNAME)`

#### Type=NS
$$
\text{Domain name} \Rightarrow \text{\textcolor{red}{N}ame \textcolor{red}{S}erver}
$$
All’interno di questo tipo di record:
- `name` → dominio (es. `foo.com`)
- `value` → nome dell’host del server di competenza di questo dominio

>[!example] `(foo.com, dns.foo.com, NS)`

#### Type=MX
$$
\text{Alias} \Rightarrow \text{\textcolor{red}{M}ail server canonical name}
$$
All’interno di questo tipo di record:
- `name` → alias di qualche nome canonico
- `value` → nome canonico del server di posta associato a `name`

>[!example] `(foo.com, mail.bar.foo.com, MX)`

#### Tipi di record
![[Pasted image 20250316175929.png]]

>[!example]
>Server di competenza per un hostname contiene:
>- un record di tipo `A` per l’hostname
>
>Server non di competenza per un dato hostname contiene:
>- un record di tipo `NS` per il dominio che include l’hostname
>- un record di tipo `A` che fornisce l’indirizzo IP del server DNS di competenza nel campo `value` del record DNS
>
>>[!question] Come funziona nella pratica?
>>1. Se un resolver DNS cerca `www.example.com`, ma il server interrogato non è autoritativo, ottiene l'indicazione di un altro server da contattare
>>2. Il resolver ripete la richiesta al server autoritativo (indicato nel record `NS`), che infine risponde con il record `A` contenente l’indirizzo IP richiesto

### Messaggi DNS
Per il protocollo DNS **domande** (query) e messaggi di **risposta** hanno entrambi lo **stesso formato**

![[Pasted image 20250316181354.png|center|500]]

Nell’intestazione del messaggio si ha:
- **identificazione** → numero di $16\text{ bit}$ per la domanda; la risposta alla domanda usa lo stesso numero (così da identificare chi nello specifico ha fatto la domanda)
- **flag**
	- domanda o risposta
	- richiesta di ricorsione
	- ricorsione disponibile
	- risposta di competenza (il server competente per il nome richiesto)
- numero di occorrenze delle quattro sezioni di tipo dati successive

Nel corpo del messaggio si ha in ordine:
- campi per il nome richiesto e il tipo di domanda (`A, MX`)
- RR nella risposta alla domanda; più RR nel caso di server replicati
- record per i server di competenza
- informazioni extra che possono essere usate (nel caso di una risposta `MX`, il campo di riposta contiene il record `MX` con il nome canonico del server di posta, mentre la sezione agguintiva contiene un record di tipo `A` con l’indirzzo IP relativo all’hostname canonico del server di posta)


> [!example]
> Immaginiamo che un client voglia risolvere `www.example.com` in un indirizzo IP.
> 
> ##### Query
>Il client invia al server DNS una richiesta con:
>- `ID`: 1234
>- `Flags`: richiesta (`QR = 0`)
>- `Numero di domande`: 1 (`www.example.com`, tipo `A`)
>- Le altre sezioni sono vuote.
> 
>##### Risposta
>Il server DNS risponde con:
>- `ID`: 1234 (lo stesso della richiesta)
>- `Flags`: risposta (`QR = 1`)
>- `Numero di domande`: 1 (`www.example.com`)
>- `Numero di RR di risposta`: 1 (`A 93.184.216.34`)
>- `Numero di RR autorevoli`: 1 (`NS ns1.example.net`)
>- `Numero di RR addizionali`: 1 (`ns1.example.net A 203.0.113.10`)
> 
>La risposta dice che:
>- `www.example.com` ha l’IP `93.184.216.34` (sezione Risposte).
>- Il server DNS autoritativo è `ns1.example.net` (sezione Competenza).
>- L'IP di `ns1.example.net` è `203.0.113.10` (sezione Informazioni Aggiuntive).

---
## Inserire record nel database DNS
Immaginiamo di aver appena avviato la nuova società “Network Stud”

>[!info]
>E’ possibile aggiungere nuovi domini al DNS contattando un registrar (aziende commerciali accreditate dall’ ICANN).
>Il registrar in cambio di un compenso verifica l’unicità del dominio richiesto e lo inserisce nel database

Registriamo il nome networkstud.it presso un *registrar* (www.registro.it) e inseriamo nel server di competenza (es. un nostro DNS server `dns1.networkstud.it`) un record tipo `A` per `www.networkstud.it` e un record tipo `MX` per `networkstud.it`
- `A` record → $\verb|www.networkstud.it|\rightarrow \verb|150.160.15.12|$ (indirizzo del server web)
- `MX` record: $\verb|networkstud.it|\rightarrow \verb|mail.networkstud.it|$ (posta elettronica)
- `CNAME` record:  $\verb|mail.networkstud.it|\rightarrow \verb|mail.google.com|$ (se usiamo Gmail)

![[Pasted image 20250316182758.png]]


