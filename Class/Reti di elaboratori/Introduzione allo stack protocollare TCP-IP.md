---
Created: 2025-03-13
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Si è fornita una panoramica della struttura e delle prestazioni di Internet, che è costituita da numerose reti di varie dimensioni interconnesse tramite opportuni dispositivi di comunicazione. Tuttavia per poter comunicare non è sufficiente assicurare questi collegamenti, ma è necessario utilizzare sia dell’hardware che del software (**hardware e software devono essere coordinati**)

>[!example]- Esempio di comunicazione
>![[Pasted image 20250313201724.png]]
>Dure interlocutori rispettano un protocollo di conversazione (interazione).
>- Si inizia con un saluto
>- Si adotta un linguaggio appropriato al livello di conoscenza
>- Si tace mentre l’altro parla
>- La conversazione si sviluppa come dialogo piuttosto che un monologo
>- Si termina con un saluto

---
## Protocollo
Un protocollo **definisce le regole** che il mittente e il destinatario, così come tutti i sistemi coinvolti, devono rispettare per essere in grado di comunicare.

In situazioni particolarmente semplici potrebbe essere sufficiente un solo protocollo, in situazioni più complesse potrebbe essere opportuno suddividere i compiti fra più **livelli** (*layer*), nel qual caso è richiesto un protocollo per ciascun livello (si parla di *layering di protolli*)

---
## Organizzazione a più livelli

>[!example]
>Anna viene trasferita e le due amiche continuano a sentirsi via posta. Poiché hanno in mente un progetto innovativo vogliono rendere sicura la conversazione mediante un meccanismo di crittografia. Il mittente di una lettera la cripta per renderla incomprensibile a un eventuale intruso, il destinatario la decripta per recuperare il messaggio originale
>
>![[Pasted image 20250313215922.png|500]]
>- Si ipotizza che le due amiche abbiano tre macchine ciascuna per portare a termine i compiti di ciascun livello
>- Supponiamo che Maria invii la prima lettera
>- Maria comunica con la macchina al terzo livello come se fosse Anna e la potesse ascoltare
>
>La strutturazione dei protocolli in livelli consente di **suddividere un compito complesso in compiti più semplici**.
>
>Si potrebbe usare una sola macchina ma cosa accadrebbe se le due amiche decidessero di cambiare la tecnica di crittografia? Usando le 3 macchine dell’esempio verrebbe sostituita solo quella intermedia (**modularizzazione**, indipendenza dei livelli)

### Principi della strutturazione a livelli
Quando è richiesta una comunicazione **bidirezionale**, ciascun livello deve essere capace di effettuare i due compiti opposti, uno per ciascuna direzione (es. crittografare, decrittografare).

