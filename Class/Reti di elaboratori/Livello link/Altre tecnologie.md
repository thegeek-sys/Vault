---
Created: 2025-05-22
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Index
- [[#Bluetooth|Bluetooth]]
	- [[#Bluetooth#Architettura piconet e scatternet|Architettura piconet e scatternet]]
	- [[#Bluetooth#Layers|Layers]]
	- [[#Bluetooth#Protocollo MAC|Protocollo MAC]]
---
## Bluetooth
La tecnologia bluetooth è una tecnologia LAN wireless progettata per connettere dispositivi con diversi funzioni (telefoni, stampanti, etc.) e diverse capacità

Ha un raggio trasmissivo di circa $10\text{ m}$ e una LAN Bluetooth è una rete ad hoc che si forma spontaneamente senza aiuto di alcuna stazione base
Si tratta però di una rete limitata e che permette solo a pochi dispositivi di far parte della rete. Ha una banda di $2.4\text{ GHz}$ ed è divisa in $79$ canali da $1\text{ MHz}$ ciascuno

La IEEE $802.15$ standard per Persona Area Network ha implementato la tecnologia Bluetooth

>[!warning]
>Poiché lavora sulla stessa ampiezza di banda delle reti LAN IEEE $802.11b$ potrebbe provocare interferenze

### Architettura piconet e scatternet
Bluetooth definisce 2 tipi di reti:
- **piconet** → rete composta al massimo di 8 dispositivi (una stazione primaria e sette secondarie che si sintonizzano con la primaria); possono esserci altre stazioni secondarie, ma in stato di *parked* (sincronizzate con la primaria ma non possono prendere parte alla comunicazione) finchè una stazione attiva non viene spostata nello stato di parked o lascia il sistema
	![[Pasted image 20250522004742.png|center|400]]
- **scatternet** → combinazione di piconet; una secondaria in una piconet può essere una primaria in un’altra piconet, passando messaggi da una rete all’altra
	![[Pasted image 20250522004850.png|center|450]]

### Layers
Bluetooth definisce uno stack protocollare diverso da TCP/IP
![[Pasted image 20250522005034.png|500]]

### Protocollo MAC
Il Bluetooth usa il TDMA con slot temporali da $625\mu s$

La stazione primaria e secondaria possono entrambe ricevere e inviare i dati, ma non contemporaneamente (half duplex). Per fare in modo che ciò accada, la primaria usa solo gli slot pari, la secondaria quelli dispari

![[Pasted image 20250522005518.png]]

Lo slot temporale è cosi utilizzato:
- trasmissione pacchetto ($\sim 366\mu s$) → include access code, header e payload
- tempo di guardia ($\sim 259\mu s$) → server per turnaround ($TX \iff RX$), salto di frequenza e sincronizzazione

Nella comunicazione con più secondarie, la primaria usa slot pari e ad ogni slot specifica chi deve trasmettere nello slot successivo
![[Pasted image 20250522005558.png]]