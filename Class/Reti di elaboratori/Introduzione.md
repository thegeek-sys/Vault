---
Created: 2025-02-28
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## La rete
La rete è composta tipicamente da due tipi di dispositivi in grado di scambiarsi informazioni:
- **dispositivi terminali** (end system)
- **dispositivi di interconnessione**

### Dispositivi terminali
I dispositivi terminali esistono di due tipi:
- **host** → macchina in genere di proprietà degli utenti e dedicata ad eseguire applicazioni
- **server** → tipicamente un computer con elevate prestazioni destinato ad eseguire programmi che forniscono servizi

### Dispositivi di interconnessione
Per far comunicare i nodi vengono usati i dispositivi di interconnessione, rigenerando/modificando il segnale. Questi sono di tre tipi:
- **router** → dispositivi che collegano una rete ad altre reti
- **switch** → collegano sistemi terminali tra loro a livello locale
- **modem** → trasformano i dati digitali del computer in dati analogici della rete (ha solo il compito di rigenerare il segnale)

---
## Collegamenti
I dispositivi di rete vengono collegati utilizzando mezzi trasmissivi cablati o wireless generalmente chiamati **link**
Questi sono di due tipi:
- cablati
- wireless

### Mezzi trasmissivi cablati
In questo caso il bit viaggia **tramite un mezzo fisico** che si trova tra il trasmittente e il ricevente. Il più interessante tra questi è la **fibra ottica**; la sua potenza infatti sta nel fatto che non subisce interferenze, il che implica un bassissimo tasso di errore

### Mezzi trasmissivi wireless
In questo caso i segnali si propagano nell’**atmosfera** e possono avere molte più interferenze (soffrono molto l’ambiente circostante) ma non richiedono l’installazione fisica di cavi

---
## Classificazione delle reti

| Scala      | Tipo                            | Esempio        |
| ---------- | ------------------------------- | -------------- |
| Prossimità | PAN (Personal Area Network)     | Bluetooth      |
| Edificio   | LAN (Local Area Network)        | WiFi, Ethernet |
| Città      | MAN (Metropolitan Area Network) | Cable, DSL     |
| Nazione    | WAN (Wide Area Network)         | Large ISP      |
| Globo      | The Internet                    | The Internet   |

---
## Reti LAN
Una rete LAN solitamente è una **rete privata che collega i sistemi terminali** in un singolo ufficio (azienda, università), per permettergli di condividere risorse. Ogni sistema terminale nella LAN ha un indirizzo che lo identifica univocamente nella rete.

Si differenziano in due tipi:
- LAN a cavo condiviso (broadcast)
- LAN a commutazione (con switch)

### LAN con cavo condiviso (broadcast)
![[Pasted image 20250228101112.png|center|400]]
Qui quando un nodo trasmette, tutti gli altri ricevono quella trasmissione ma solo il destinatario la elabora.
In questo caso solo un terminale alla volta può trasmettere

### LAN con switch di interconnessione
![[Pasted image 20250228101606.png|center|350]]
Qui ogni dispositivo in rete è direttamente collegato allo switch, il quale è in grado di riconoscere l’indirizzo di destinazione e di inviare il pacchetto al solo destinatario (senza inviarlo agli altri).
In questo caso più host alla volta possono trasmettere contemporaneamente

---
## Rete WAN
Le reti WAN permettono l’**interconnessione di dispositivi in grado di comunicare** (interconnette switch, router, modem) e tipicamente è gestita da un internet service provider (*ISP*) che fornisce servizi alle varie LAN

Esistono di due tipi:
- WAN punto-punto
- WAN a commutazione (switched)

>[!example]- WAN punto-punto
>
![[Pasted image 20250228102058.png|center|450]]
Collega due mezzi di comunicazione tramite un mezzo trasmissivo (cavo o wireless)

>[!example]- WAN a commutazione
>![[Pasted image 20250228102209.png|center|450]]
>Compone una rete con più di due punti di terminazione e viene utilizzata nelle dorsali di internet

###  Internetwork composta da due LAN e una WAN punto-punto
Oggigiorno tipicamente LAN e WAN sono connesse tra loro per formare una internetwork (o internet).
Ad esempio un’azienda che ha due uffici in città differenti ha una rete LAN in ogni singolo ufficio (per far comunicare gli impiegati tra loro) e affitta una WAN punto-punto da un ISP realizzando un internet privato
![[Pasted image 20250228102755.png|center|550]]

### Rete eterogenea composta da quattro WAN e tre LAN
![[Pasted image 20250228102901.png|center|500]]

### GARR
La rete GARR interconnette ad altissima capacità università, centri di
ricerca, biblioteche, musei, scuole e altri luoghi in cui si fa istruzione, scienza, cultura e innovazione su tutto il territorio nazionale. È un’infrastruttura in fibra ottica che utilizza le più avanzate tecnologie di comunicazione e si sviluppa su circa 15.000 km tra collegamenti di dorsale e di accesso.
Oggi la capacità delle singole tratte della dorsale arriva a 100 Gbps, mentre quella dei collegamenti di accesso può raggiungere i 40 Gbps in base alle necessità di banda della sede. Grazie alla grande scalabilità delle tecnologie utilizzate, queste capacità possono evolvere facilmente insieme alle necessità degli utenti. È prossimo infatti il primo collegamento utente a 100 Gbps.

---
## Commutazione (switching)
Una internet è una combinazione di link e dispositivi capaci di scambiarsi informazioni. In particolare i sistemi terminali appartenenti alla rete comunicano tra loro per mezzo di dispositivi come switch e router che si trovano nel percorso tra i sistemi sorgente e destinazione

In base al funzionamento degli switch esistono due tipi di reti:
- **reti a commutazione di circuito**
- **reti a commutazione di pacchetto**

### Reti a commutazione di circuito
Tra due dispositivi è sempre disponibile un collegamento dedicato, chiamato circuito, usato per l’intera comunicazione.
Vengono riservate le risorse necessarie per la comunicazione (banda, etc.) e le informazioni riguardanti il circuito vengono mantenute all’interno della rete (simile al sistema telefonico)

![[Pasted image 20250228103801.png|center|400]]
Anche se esistono più percorsi tra due dispositivi in comunicazione, solo uno di questi verrà usato per l’intera comunicazione

#### Efficienza
La sua efficienza sta nel fatto che nel momento in cui viene stabilito un circuito viene garantita una certa capacità
![[Pasted image 20250228104052.png|400]]

Infatti  le risorse di rete vengono suddivise in “pezzi”, i quali vengono allocati ai vari collegamenti. Questa tipo di suddivisione può essere: della frequenza o del tempo

#### FDM e TDM
>[!info] FDM → Frequency Division Multiplexing
>![[Pasted image 20250228104803.png|450]]

>[!info] TDM → Time Division Multiplexing
>![[Pasted image 20250228104847.png|450]]

### Rete a commutazione di pacchetto (store and forward)
In questo caso non viene riservato nessun tipo di risorsa, ma quello che si va a “suddividere” sono i dati. Per mandare ad esempio un video viene diviso in pacchetti per poi essere condiviso (non si ha una comunicazione continua). I dati vengono quindi trasmessi in blocchi, o **pacchetti**.

Il mittente dunque spedisce i suoi pacchetti al router che li inserisce in una coda (*store*) e uno alla volta li inoltra (*forward*).

![[Pasted image 20250228105203.png|500]]

Ogni nodo deve ricevere un pacchetto per intero e poi lo invia all’host ricevente (in realtà i router hanno una coda per singolo host). In questo esempio hanno 5 code di entrata/uscita (può ricevere anche dall’altro router)

![[Pasted image 20250228105456.png|450]]

Inoltre **ogni pacchetto viaggia indipendentemente dagli altri**, per questo motivo non necessariamente se vengono mandati più pacchetti da uno stesso mittente ad uno stesso ricevente questi arrivano nello stesso momento (possono prendere percorsi diversi).

Risulta essere molto più flessibile in termine di prestazioni rispetto alla commutazione di circuito, infatti se ci sta un solo dispositivo può trasmettere a velocità piena e solo se ci stanno più host viene suddivisa la banda

---
## Internet
Si è detto che una internet (con i minuscola) è costituita da due o più reti interconnesse. L’internet più famosa è chiamata Internet (I maiuscola) ed è composta da migliaia di reti interconnesse (è una rete a commutazione di pacchetto)

![[Pasted image 20250228105858.png|500]]

---
## L’accesso a Internet
Si hanno tre modi per accedere ad Internet:
- accesso via **rete telefonica** → servizio dial-up (via modem), servizio DSL (digital subscriber line)
- accesso tramite **reti wireless** → WiFi, cellulare
- **collegamento diretto** → aziende di grandi dimensioni possono divenire ISP locali, affittando delle reti WAN da un operatore

### Accesso via rete telefonica
E’ possibile collegarsi a Internet modificando la linea telefonica fra la sede del dispositivo che vuole connettersi (casa, azienda, etc.) e la centrale telefonica con una WAN punto-punto
#### Servizio dial-up
Consiste nell’inserire sulla linea telefonica un modem (modulatore-demodulatore) che converte i dati digitali (del computer) 