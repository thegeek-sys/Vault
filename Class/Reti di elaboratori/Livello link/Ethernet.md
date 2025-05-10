---
Created: 2025-05-10
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Introduction
Nel 1985 la IEEE Computer Society iniziò un progetto chiamato **Progetto 802** con l’obiettivo di definire uno standard per l’interconnessione tra dispositivi di produttori differenti così da poter definire le funzioni del livello fisico e di collegamento dei protocolli LAN

---
## Standard IEEE 802
La IEEE ha prodotto diversi standard per le LAN, collettivamente noti come IEEE 802. Essi includono gli standard per:
- **specifiche generali** del progetto → $802.1$
- **logical link control** (LLC) → $802.2$ (rilevazione errori, controllo flusso, parte del framing)
- **CSMA/CD** → $802.3$
- **token bus** → $802.4$ (destinato a LAN per automazione industriale)
- **token ring** → $802.5$
- **DQBD** → $802.6$ (destinato alle MAN)
- **WLAN** → $802.11$

I vari standard differiscono a livello fisico e nel sottolivello MAC, ma sono compatibili a livello data link

![[Pasted image 20250510222845.png]]

---
## Ethernet
L’**Ethernet** detiene una posizione dominante nel mercato delle LAN cablate. E’ stato infatti la prima LAN ad alta velocità con vasta diffusione in quanto più semplice e meno costosa di token ring, FDDI e ATM, e riesce a stare al passo dei tempi con il tasso trasmissivo

Tutte le stazioni che fanno parte di una ethernet sono dotate di una Network Interface Card (NIC, o scheda di rete). La NIC fornisce un indirizzo di rete di livello di collegamento. Gli indirizzi vengono trasmessi da sinistra verso destra, byte per byte, ma per ciascun byte il bit meno significativo viene inviato per primo e quello più significativo per ultimo

![[Pasted image 20250510223223.png]]

### Ethernet standard
L’Ethernet standard supporta fino a $10\text{ Mbps}$. I frame sono così formati
![[Pasted image 20250510224214.png]]
- **preambolo** ($7$ byte) → sette byte hanno i bit $10101010$ e server per “attivare” le network interface card dei riceventi e sincronizzare i loro orologi con quello del trasmittente (fa parte dell’header del livello fisico)
- **SFD** (Start Frame Delimiter, $1$ byte) → $10101011$, flag che definisce l’inizio del frame (ultima possibilità di sincronizzazione); gli ultimi due bit $11$ indicano che inizia l’header MAC
- **indirizzo sorgente e destinazione** ($6$ byte) → quando una NIC riceve un pacchetto contenente il proprio indirizzo di destinazione o l’indirizzo boradcast (es. pacchetto APR), trasferisce il contenuto del campo dati del pacchetto al livello di rete (i pacchetti con altri indirizzi MAC vengono ignorati)
- **tipo** ($2$ byte) → utile per multiplexing/demultiplexing, specifica che tipo di protocollo è stato incapsulato nel campo dati (IP, ARP, OSPF, etc.)
- **dati** (da $46$ a $1500$ byte) → contiene il datagramma di rete; se il datagramma è inferiore alla dimensione minima il campo viene *stuffed* con degli zeri fino a raggiungere quel valore
- **CNR** → consente alla NIC ricevente di rilevare la presenza di un errore nei bit sul campo indirizzo, tipo e dati

L’Ethernet standard è un protocollo senza connessione, e dunque è non affidabile (la NIC ricevente non invia un riscontro)

La lunghezza minima del frame è di $64$ byte di cui $18$ per l’intestazione e trailer e $46$ dei dati provenienti dal livello superiore (se inferiore si esegue il padding, bit nulli di riempimento). E’ necessaria per il corretto funzionamento del CSMA/CD.
La lunghezza massima del frame è di $1518$ byte, di cui $18$ di intestazione e trailer e $1500$ di dati. E’ necessaria per evitare che una stazione possa monopolizzare il mezzo e per ragioni storiche (la memoria era molto costosa e questa restrizione permetteva di ridurre la memoria necessaria nei buffer dei dispositivi)

