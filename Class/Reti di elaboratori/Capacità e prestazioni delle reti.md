---
Created: 2025-03-02
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Rete di acceso|Rete di acceso]]
- [[#Struttura di Internet|Struttura di Internet]]
- [[#Capacità e prestazioni|Capacità e prestazioni]]
- [[#Bandwidth e bit rate|Bandwidth e bit rate]]
- [[#Throughput|Throughput]]
	- [[#Throughput#Un percorso attraverso il backbone Internet|Un percorso attraverso il backbone Internet]]
	- [[#Throughput#Effetto del throughput nei link condivisi|Effetto del throughput nei link condivisi]]
- [[#Latenza (delay)|Latenza (delay)]]
	- [[#Latenza (delay)#Ritardo di nodo|Ritardo di nodo]]
	- [[#Latenza (delay)#Ritardo di accodamento|Ritardo di accodamento]]
- [[#Perdita di pacchetti (packet loss)|Perdita di pacchetti (packet loss)]]
- [[#Ritardi e percorsi in Internet|Ritardi e percorsi in Internet]]
- [[#Prodotto $\text{rate}\cdot \text{ritardo}$|Prodotto rate*ritardo]]
---
## Introduction
L’internet può essere interpretato come una **rete di reti** composta da reti di accesso e backbone Internet

---
## Rete di acceso
![[Pasted image 20250302164539.png|center|350]]

La rete di accesso è composta da tutti i dispositivi che ci danno la possibilità di raggiungere il primo router del backbone (*edge router*); dove per **backbone** si intende la rete di tutti i router (composta solamente da router e dai loro collegamenti)

Il backbone Internet può essere di due tip (come già precedentemente detto):
- a **commutazione di circuito** → circuito dedicato per l’intera durata della sessione
- a **commutazione di pacchetto** → i messaggi di una sessione utilizzano le risorse su richiesta

---
## Struttura di Internet
La struttura di Internet è fondamentalmente gerarchica:
- ISP di primo livello → hanno la copertura più vasta, fornendo servizi a livello nazionale/internazionale. Un esempio è la **SEA-ME-WE 6**, il sistema South East Asia-Middle East-West Europe 6, che collega Singapore alla Francia attraverso cavi sottomarini e terrestri
- ISP di secondo livello → risulta essere ben più piccolo rispetto a quelli di primo livello provvedendo alla distribuzione nazionale o distrettuale, sfruttando i servizi disposti dagli ISP di primo livello
- ISP di terzo livello e ISP livello → chiamate *last hop network*, sono le reti più vicine ai sistemi terminali

![[Pasted image 20250302165735.png|center|550]]

Uno dei problemi principali delle reti è quello di trovare il percorso da seguire per arrivare a destinazione. Questo problema è denominato **routing** (instradamento)

---
## Capacità e prestazioni
Un argomento di cui si parla spesso nell’ambito delle reti è la velocità della rete o di un collegamento, ovvero quanto velocemente si riesce a trasmettere e ricevere i dati (seppur il concetto di velocità sia più ampio e coinvolga più fattori).

Nel caso di una rete a commutazione di pacchetto, le metriche ne determinano le prestazioni e si misurano in:
- Ampiezza di banda → bandwidth e bit rate
- Throughput
- Latenza
- Perdita di pacchetti

---
## Bandwidth e bit rate
Con il termine **ampiezza di banda** si indicano due concetti diversi ma strettamente legati: la **bandwidth** e il **bit rate**

La **bandwidth** è una caratteristica del mezzo trasmissivo, si misura i Hz e rappresenta l’**ampiezza dell’intervallo di frequenza utilizzato dal mezzo trasmissivo**.
Maggiore e la ampiezza di banda maggiore è la quantità di informazione che può essere veicolata attraverso il mezzo.

Il **bit rate** indica quanti bit posso trasmettere per unità di tempo; la sua unità di misura è il bps (bit per second)

Il bit rate dipende sia dalla banda (maggiore è la banda maggiore è il bit rate) che dalla specifica tecnica di trasmissione

---
## Throughput
Throughput è una misura che ci dice quanto **effettivamente** (in contrapposizione a nominalmente) una rete riesce ad inviare dati. Somiglia al bit rate ma tendenzialmente gli è minore o uguale (il bit rate è la potenziale velocità di un link)

>[!example]
>Una strada è progettata per far transitare $1000$ auto al minuto da un punto all’altro. Se c’è traffico, tale cifra può essere ridotta a $100$. Il rate è $1000$ auto al minuti, il troughput è di $100$ auto al minuto

In un percorso da una sorgente ad una destinazione un pacchetto può passare attraverso numerosi link, ognuno con throughtput diverso

>[!question] Come si determina il thoughput dell’intero percorso (end to end)?
>![[Pasted image 20250303094542.png|500]]
>In questo primo caso il throughput è $R_{s}$
>
>![[Pasted image 20250303094617.png|500]]
>In questo caso invece il throughput è $R_{c}$

>[!example] Throughput su un percorso di tre link
>![[Pasted image 20250303094845.png|440]]
>Il throughput dei dati per il percorso è $100 \text{ kbps}$

In generale in un percorso con $n$ link in serie abbiamo:
$$
\text{Throughput = min}\{T_{1},T_{2},\dots,T_{n}\}
$$

### Un percorso attraverso il backbone Internet
La situazione reale in Internet è che i dati normalmente passano attraverso due reti di accesso e la dorsale Internet.

![[Pasted image 20250303095214.png]]

La dorsale ha un throughput molto alto (nell’ordine dei gigabit al secondo), quindi il thoughput viene definito come il minimo tra i due link di accesso che collegano la sorgente e la destinazione alla dorsale

Nell’esempio il thoughput è il minimo tra $T_{1}$ e $T_{2}$. Se $T_{1}$ è $100 \text{ Mbps}$ (Fast Ethernet LAN) e $T_{2}$ è $40 \text{ kbps}$ (linea telefonica commutata), il troughput è $40 \text{ kbps}$

### Effetto del throughput nei link condivisi
ùIl link tra due router non è sempre dedicato a un flusso di dati, ma raccoglie il flusso da varie sorgenti e/o lo distribuisce a varie destinazioni. Il **rate del link tra i due router è condiviso tra i flussi di dati**.

>[!example]
>La velocità del link principale nel calcolo del troughput è solo  $200\text{ kbps}$ in quanto il link è condiviso. Il troughput end to end è $200\text{ kbps}$
>![[Pasted image 20250303095710.png|400]]

---
## Latenza (delay)
La **latenza** è il tempo impiegato affinché un pacchetto arrivi completamente a destinazione del momento in cui il primo bit parte dalla sorgente

Nella commutazione di pacchetto i pacchetti si accodano nei buffer dei router, ma se il tasso di arrivo dei pacchetti sul collegamento eccede la capacità del collegamento di evaderli, i pacchetti si accordano in attesa del proprio turno

![[Pasted image 20250303100144.png|450]]

Esistono quattro tipi di case di ritardo per i pacchetti:
- ritardo di **elaborazione** (*processing delay*)
	- controllo errori sui bit (in questo caso viene scartato)
	- determinazione del canale di uscita
	- tempo della recezione della porta di input alla consegna alla porta di output
- ritardo di **accodamento** (*queueing delay*)
	- attesa di trasmissione → sia nella coda di input che nella coda di output in base al grado di congestione del router (devono attendere che gli altri pacchetti entrino/escano)
	- varia da pacchetto a pacchetto → infatti potrebbe succedere che la coda verso un router sia piena mentre la coda verso un altro router è vuota (quello verso il secondo router verrà inviato prima)
- ritardo di **trasmissione** (*transmission delay*)
	- il ritardo di trasmissione dipende dal rate del collegamento e dalla lunghezza del pacchetto (bit che devo trasmettere). 
		Quindi il ritardo di trasmissione è dato da $L/R$ dove $R$ è il rate del collegamento (in $bps$) e $L$ è la lunghezza del pacchetto (in $bit$)
- ritardo di **propagazione** (*propagation delay*)
	- il tempo impiegato dal pacchetto, una volta immesso sul canale, per raggiungere (propagarsi) il prossimo router/destinazione.
		Questo è dato dalla $d/s$ dove $d$ è la lunghezza del canale (distanza che devo percorrere) e $s$ è la velocità di propagazione (velocità della luce, valida per il singolo bit)

>[!example] Analogia del casello autostradale #1
>![[Pasted image 20250303101909.png|500]]
>Le automobili viaggiano (ossia “si propagano”) alla velocità di $100\text{ km/h}$. Il casello serve (ossia “trasmette“) un’auto ogni $12 \text{ sec}$ (rate)
>
>Quanto occorre perché le $10$ auto abbiano superato il secondo casello?
>Tempo richiesto al casello per trasmettere l’intera colonna sull’autostrada:
>$$\frac{10}{5\text{ auto/min}}=2\text{ min}$$
>Tempo richiesto a un’auto per viaggiare dall’uscita di un casello fino al casello successivo:
>$$\frac{100\text{ km}}{100 \text{ km/h}}=1\text{ h}$$
>
>Il tempo che intercorre da quando l’intera colonna di vetture di trova di fronte al casello di partenza fino al momento in cui raggiunge quello successivo è la somma del ritardo di trasmissione e del ritardo di propagazione ovvero $62 \text{ minuti}$

>[!example] Analogia del casello autostradale #2
>![[Pasted image 20250303101909.png|500]]
>Le auto ora “si propagano” alla velocità di $1000\text{ km/h}$
>Al casello adesso occorre $1\text{ min}$ per servire ciascuna auto
>
>Le prime auto arriveranno al secondo casello prima che le ultime auto della colonna lascino il primo? Si
>Dopo $7$ minuti, la prima auto sarà al secondo casello, e tre auto saranno ancora in coda davanti al primo casello
>
>>[!warning]
>>l primo bit di un pacchetto può arrivare al secondo router prima che il pacchetto sia stato interamente trasmesso dal primo router

### Ritardo di nodo
$$
d_{\text{nodo 1}}=d_{\text{proc}}+d_{\text{queue}}+d_{\text{trans}}+d_{\text{prop}}
$$
- $d_{\text{trans}}$ → ritardo di trasmissione (significativo sui collegamenti a bassa velocità)
- $d_{\text{prop}}$ → ritardo di propagazione (da pochi microsecondi a centinaia di millisecondi)
- $d_{\text{proc}}$ → ritardo di elaborazione (in genere pochi microsecondi, o anche meno)
- $d_{\text{queue}}$ → ritardo di accodamento (dipende dalla congestione)

### Ritardo di accodamento
Il ritardo di accodamento, come già detto, può variare da pacchetto a pacchetto e dipende dal tasso di arrivo, dal rate e dalla lunghezza dei pacchetti
- $R=\text{rate di trasmissione (bps)}$
- $L=\text{lunghezza del pacchetto (bit)}$
- $a = \text{tasso medio di arrivo dei pacchetti (pkt/s)}$
$$
\frac{La}{R}=\text{intensità di traffico}
$$
![[Pasted image 20250303104413.png|center]]

Se:
- $La/R \sim 0$ → poco ritardo
- $La/R \rightarrow 1$ → il ritardo si fa consistente
- $La/R>1$ → più “lavoro” in arrivo di quanto possa essere effettivamente svolto

---
## Perdita di pacchetti (packet loss)
Se una coda (detta anche buffer) ha capacità finita, **quando il pacchetto trova la coda piena, viene scartato** (e quindi va perso).
Il pacchetto perso può essere ritrasmesso dal nodo precedente, del sistema terminale che lo ha generato o non essere ritrasmesso affatto

![[Pasted image 20250303104710.png|400]]

---
## Ritardi e percorsi in Internet
Ma cosa significano effettivamente packet delay e loss nella “vera” Internet?
`traceroute` è il programma diagnostico che fornisce una misura del ritardo dalla sorgente a tutti i router lungo il percorso Internet punto-punto verso la destinazione.

![[Pasted image 20250303105053.png|400]]
In particolare invia gruppi di tre pacchetti, dove ogni gruppo ha tempo di vita incrementale (da $1$ a $n$, massimo valore $30$) che raggiungeranno il router ($i=1,\dots,n$) sul percorso verso la destinazione. Il router $i$ restituirà i pacchetti al mittente e il mittente calcola l’intervallo tra trasmissione e risposta

L’output presenta 6 colonne:
1. Il numero di router sulla rotta
2. Nome del router
3. Indirizzo del router
4. Tempo di andata e ritorno 1° pacchetto
5. Tempo di andata e ritorno 2° pacchetto
6. Tempo di andata e ritorno 3° pacchetto

Il tempo di andata e ritorno (**round trip time** - *RTT*) include i 4 ritardi visti precedentemente. Può accadere (se c’è congestione o se si segue un percorso diverso) che il RTT del router $n$ sia maggiore del RTT del router $n+1$ a causa dei ritardi di accodamento (dipendono dalla stato attuale della rete). Se la sorgente non riceve risposta da un router intermedio (o ne riceve meno di 3) allora pone un asterisco al posto del tempo RTT

>[!example] `traceroute` da `gaia.cs.umass.edu` a `www.eurecom.fr`
>![[Pasted image 20250303105409.png]]
>Ogni riga di `traceroute` rappresenta un router che viene attraversato

---
## Prodotto $\text{rate}\cdot \text{ritardo}$
Il prodotto $\text{rate}\cdot \text{ritardo}$ rappresenta il **massimo numero di bit che possono trovarsi sul canale**.

>[!example]
>Supponiamo di avere un link con rate di $1\text{ bps}$ e un ritardo di $5 \text{ sec}$
>![[Pasted image 20250303105915.png|300]]
>Ciò vuol dire dunque che non possono esserci più di $5\text{ bit}$ contemporaneamente sul link

Possiamo pensare al link tra due punti come a un tubo. La sezione trasversale del tubo rappresenta il rate e la lunghezza rappresenta il ritardo. Possiamo dire che il volume del tubo definisce il prodotto $\text{rate-ritardo}$
![[Pasted image 20250303110147.png|460]]

---
## Esercizio
>[!question]
>- Quanto tempo impiega un pacchetto di $1000\text{ byte}$ byte per propagarsi su un collegamento di $2500\text{ km}$, con velocità di propagazione pari a $2,5\text{ x }10^8 \text{ m/s}$ e rate di $2 \text{ Mbps}$
>- Questo ritardo dipende dalla lunghezza del pacchetto?
>- Calcolare il ritardo di trasmissione



$$
2\text{ Mbps}=2.000.000\text{ Bps}
$$
$$
\frac{1000\text{ byte}}{2000000\text{ Bps}}=0,5\text{ sec}
$$
$$
2500\text{ km}=2.500.000\text{ m}
$$
$$
\frac{2500000\text{ m}}{2,5\cdot10^8\text{ m/s}}=0,01\text{ sec}
$$
$$
(0,5+0,01)\text{ sec}=0,51\text{ sec}
$$
No
$$
\frac{1000\text{ byte}}{2000\text{ Bps}}=0,5\text{ sec}
$$