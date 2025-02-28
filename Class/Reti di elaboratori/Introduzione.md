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
Qui quando un nodo trasmette, tutti gli altri ricevono quella trasmissione ma solo il destinatario la elabora