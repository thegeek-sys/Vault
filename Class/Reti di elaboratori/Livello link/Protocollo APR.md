---
Created: 2025-05-10
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Indirizzi MAC
Negli indirizzi IP a 32 bit l’indirizzo riguarda il livello di rete e servono ad individuare con esattezza i punti di Internet dove sono connessi gli host sorgente e destinazione. Gli indirizzi IP sorgente e destinazione definiscono le estremità della rete ma non dicono attraverso quali collegamenti deve passare il datagramma

![[Pasted image 20250510214026.png|center|600]]

L'**indirizzo MAC** (o LAN, fisico o Ethernet) è un indirizzo a 48 bit (6 byte, rappresentati in esadecimali) utilizzato a livello di collegamento: quando un datagramma passa dal livello di rete al livello di collegamento, viene incapsulato in un frame che contiene un'intestazione con gli indirizzi di collegamento della sorgente e della destinazione del frame (non del datagramma)

Ciascun adattatore di una LAN ha un indirizzo MAC univoco

![[Pasted image 20250510214115.png]]

La IEEE sovrintende alla gestione degli indirizzi MAC. Quando una società vuole costruire degli adattatori, compra un blocco di spazio di indirizzi (unicità degli indirizzi).
Gli indirizzi MAC hanno una struttura non gerarchica. Ciò rendere possibile spostare una scheda LAN da una LAN ad un’altra, invece gli indirizzi IP hanno una struttura gerarchica e dunque devono essere aggiornati se spostati (dipendono dalla sottorete IP cui il nodo è collegato)

>[!question] Come vengono determinati gli indirizzi di collegamento dalla sorgente alla destinazione?
>**Address Resolution Protocol** (*APR*)

---
## Protocollo APR
L’**Address Resolution Protocol** (*APR*) è il protocollo utilizzato per tradurre un indirizzo IP in un indirizzo MAC

Ogni nodo IP (host, router) nella LAN ha una **tabella APR**, la quale contiene la corrispondenza tra indirizzi IP e MAC

```
<Indirizzo IP; Indirizzo MAC; TTL>
```