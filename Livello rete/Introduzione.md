---
Created: 2025-04-11
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
---
## Pila di protocolli internet
- **applicazione** → di supporto alle applicazioni di rete
	- FTP, SMTP, HTTP
- **trasporto** → trasferimento dei messaggi a livello di applicazione tra il modulo client e server di un’applicazione
	- TCP, UDP
- **rete** → instradamento dei datagrammi dall’origine al destinatario
	- IP, protocolli di instradamento
- **link** → instradamento dei datagrammi attraverso una serie di commutatori di pacchetto
	- PPP, Ethernet
- **fisico** → trasferimento dei singoli bit

>[!example] Esempio
>![[Pasted image 20250411094148.png|center|550]]
>
>- livello di trasporto → comunicazione tra processi
>- livello di rete → comunicazione tra host
>
>Il livello di rete di $\text{H1}$ prende i segmenti dal livello di trasporto, li incapsula in un datagramma e li tramette al router più vicino
>Il livello di rete di $\text{H2}$ riceve i datagrammi da $\text{R7}$, estrae i segmenti e li consegna al 

