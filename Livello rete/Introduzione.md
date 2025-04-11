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
>Il livello di rete di $\text{H1}$ prende i segmenti dal livello di trasporto, li incapsula in un datagramma e li tramette al router più vicino. Il livello di rete di $\text{H2}$ riceve i datagrammi da $\text{R7}$, estrae i segmenti e li consegna al livello di trasporto
>Il livello di rete dei nodi intermedi inoltra verso il prossimo router

---
## Funzioni chiave a livello di rete
Il livello svolge fondamentalmente due funzioni:
- **instradamento** (*routing*)
- **inoltro** (*forwarding*)

Con l’**instradamento** si determina il percorso seguito dai pacchetti dall’origine alla destinazione (crea i percorsi). Con l’**inoltro** si trasferiscono i pacchetti dall’input di un router all’output del router appropriato (utilizza i percorsi creati dal routing)

>[!hint]
>Gli algoritmi di routing creano le tabelle di routing che vengono usate per il forwarding

### Routing e forwarding
In sintesi si può dire quindi che il routing algorithm crea la **forwarding table** (determina i valori inseriti nella tabella), ovvero una tabella che specifica quale collegamento di uscita bisogna prendere per raggiungere la destinazione

>[!warning]
>Ogni router ha la propria forwarding table

![[Pasted image 20250411095158.png|450]]

---
## Switch e router
Il **packet switch** (commutatore di pacchetto) è un dispositivo che si occupa del trasferimento dall’interfaccia di ingresso a quella di uscita in base al valore del campo dell’intestazione del pacchetto

