---
Created: 2025-05-10
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Index
- [[#Indirizzi MAC|Indirizzi MAC]]
- [[#Protocollo ARP|Protocollo ARP]]
	- [[#Protocollo ARP#Formato del pacchetto ARP|Formato del pacchetto ARP]]
- [[#Indirizzamento|Indirizzamento]]
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
>**Address Resolution Protocol** (*ARP*)

---
## Protocollo ARP
L’**Address Resolution Protocol** (*ARP*) è il protocollo utilizzato per tradurre un indirizzo IP in un indirizzo MAC

![[Pasted image 20250510215109.png|450]]

Ogni nodo IP (host, router) nella LAN ha una **tabella ARP**, la quale contiene la corrispondenza tra indirizzi IP e MAC

```
<Indirizzo IP; Indirizzo MAC; TTL>
```

![[Pasted image 20250510215226.png|450]]

In questo caso il TTL è un valore in unità di tempo che indica quando bisognerà eliminare una data voce nella tabella (tipicamente 20 min)

>[!example] Tabella APR
>![[Pasted image 20250510215202.png]]

>[!example] Protocollo APR nella stessa sottorete
>$A$ vuole inviare un datagramma a $B$, e l’indirizzo MAC di $B$ non è nella tabella ARP di $A$
>
>$A$ trasmette in un pacchetto broadcast ($FF-FF-FF-FF-FF-FF$) il messaggio di richiesta APR contenente l’indirizzo IP di $B$
>
>![[Pasted image 20250510215451.png]]
>
>Ogni nodo della rete riceve ed elabora il pacchetto di richiesta ARP ma solo il nodo con l’indirizzo IP specificato risponde
>
>$B$ riceve il pacchetto APR e risponde ad $A$ comunicandogli il proprio indirizzo MAC. Il frame viene inviato all’indirizzo MAC di $A$ (in unicast)
>
>Il messaggio di richiesta APR è inviato in un pacchetto broadcast mentre il messaggio di risposta APR è inviato in un pacchetto standard. Dunque la tabella APR di un nodo si costruisce automaticamente e non deve essere configurata dall’amministratore del sistema (plug and play)
>
>![[Pasted image 20250510220416.png]]

### Formato del pacchetto ARP
I pacchetti APR vengono incapsulati direttamente all’interno di frame di livello di collegamento

![[Pasted image 20250510220522.png]]

>[!example]
>![[Pasted image 20250510220554.png]]

---
## Indirizzamento
In ognuno dei 4 livelli dello stack protocollare TCP/IP è presente l’indirizzamento fatta eccezione per il livello fisico. Infatti il livello fisico si occupa solo di trasferire i singoli bit che vengono spediti in broadcast nel mezzo trasmissivo e ricevuti da tutti i nodi che sono collegati

>[!example] Invio verso un nodo esterno alla sottorete
>![[Pasted image 20250510221120.png]]
>>[!example] Flusso di pacchetti alla sorgente ($A$)
>>![[Pasted image 20250510221231.png]]
>>>[!example] Attività nel router $R 1$
>>>![[Pasted image 20250510221309.png]]
>
>>[!example] Flusso di pacchetti alla destinazione ($B$)
>>![[Pasted image 20250510221356.png|300]]



