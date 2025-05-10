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

![[Pasted image 20250510223223.png]]

### Ethernet standard
L’Ethernet standard 