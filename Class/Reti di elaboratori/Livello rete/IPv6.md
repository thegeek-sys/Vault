---
Created: 2025-05-09
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Formato datagramma IPv6|Formato datagramma IPv6]]
- [[#Dual stack|Dual stack]]
- [[#Tunneling|Tunneling]]
- [[#Traduzione dell’intestazione|Traduzione dell’intestazione]]
---
## Introduction
L’**IPv6** (o IP new generation) è nato con lo scopo di:
- aumentare lo spazio di indirizzi rispetto a IPv4 → indirizzi lunghi 128 bit
- ridisegnare il formato dei datagrammi
- rivedere protocolli ausiliari come ICMP

I cambiamenti in particolare riguardano:
- nuovo formato header IP
- nuove opzioni
- possibilità di estensione
- opzioni di sicurezza
- maggiore efficienza (no frammentazione nei nodi intermedi, etichette di flusso per traffico audio/video)

L’adozione di IPv6 però è lenta a causa di altre soluzioni più immediate per tamponare la crescente richiesta di indirizzi IP, come l’indirizzamento senza classi, il DHCP o il NAT

![[Pasted image 20250509124750.png]]

---
## Formato datagramma IPv6
Datagramma IPv6
![[Pasted image 20250509124535.png|500]]

Intestazione di base
![[Pasted image 20250509124607.png]]

---
## Dual stack
Durante la transizione gli host devono avere una doppia pila di protocolli per la comunicazione in rete: IPv4 e IPv6

![[Pasted image 20250509124958.png]]

	Per determinare quale versione utilizzare per inviare un pacchetto a una destinazione l’host sorgente interroga il DNS e si usa il protocollo relativo all’indirizzo ritornato (se ritorna un indirizzo IPv4 o IPv6)

---
## Tunneling
Il **tunneling** è la tecnica da utilizzare quando due host IPv6 che vogliono comunicare devono passare attraverso una regione IPv4.

In particolare si incapsula il datagramma IPv6 nel payload di un datagramma IPv4 e si inseriscono come IP sorgente e destinazione gli estremi del tunnel

![[Pasted image 20250509125553.png]]

---
## Traduzione dell’intestazione
Quando un mittente IPv6 comunica con un destinatario IPv4 è necessario effettuare una traduzione del datagramma prima che arrivi a destinazione
![[Pasted image 20250509125705.png]]


![[Pasted image 20250509125747.png]]