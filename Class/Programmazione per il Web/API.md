---
Class: "[[Programmazione per il Web]]"
Related:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Elementi di un contratto|Elementi di un contratto]]
- [[#Benefici|Benefici]]
- [[#Locali e remote|Locali e remote]]
	- [[#Locali e remote#API locali|API locali]]
	- [[#Locali e remote#API remote (Web API)|API remote (Web API)]]
- [[#Private vs. pubbliche|Private vs. pubbliche]]
- [[#Documentazione e definizione|Documentazione e definizione]]
	- [[#Documentazione e definizione#OAS - OpenAPI Specification|OAS - OpenAPI Specification]]
---
## Introduction
Un’**API** (*Application Programming Interface*) è la definizione delle interazioni consentite tra due parti di un software.

L’API funge da **contratto**, specificando come un pezzo di codice o un servizio può interagire con un altro

Nel creare una API è importante garantire **stabilità dell’interfaccia**, in modo tale da evitare che i cambiamenti possano compromettere la compatibilità con i client esistenti. Per questo motivo esistono dei marcatori di stato:
- **`beta`** → indica le parti che potrebbero cambiare poiché non ancora stabili
- **`deprecated`** → indica che le parti verranno rimosse o non saranno più supportate in 

---
## Elementi di un contratto
L’API definisce in dettaglio il “contratto” di interazione tra il consumer (il client) e il provider (il servizio). Essa soecifica:
- le richieste possibili
- i parametri delle richieste
- i valori di ritorno
- qualsiasi formato di dato richiesto (es. JSON, XML, YAML)

---
## Benefici
L’adozione di un’API porta vantaggi fondamentali nell’architettura software:
- **interfaccia esplicita** → definisce chiaramente le aspettative e le modalità di interazione
- **contratto infrangibile** → stabilisce un insieme di regole che entrambe le parti devono rispettare
- **information hiding** → la logica interna del provider (come i dati vengono processati o archiviati) rimane nascosta al consumer (il client deve solo conoscere l’interfaccia)

---
## Locali e remote
Esistono diverse categorie di API in base alla loro posizione e funzione
### API locali
- API per i linguaggi di programmazione
- API del sistema operativo
- API delle librerie software
- API hardware
### API remote (Web API)
Le API remote sono interfacce di programmazione basate su protocolli di rete (tipicamente HTTP), come le API RESTful

---
## Private vs. pubbliche
Le API private e pubbliche si distinguono in base all’accessibilità

| Categoria         | Descrizione                                                 | Restrizioni                                                                 |
| ----------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| **API private**   | destinate all’uso interno di un’azienda o un sistema chiuso | l’access è limitato ai componenti interni                                   |
| **API pubbliche** | disponibili per l’uso da parte del pubblico                 | l’accesso può essere limitato solo ad alcuni utenti tramite le *API Tokens* |

---
## Documentazione e definizione
Per stabilire il contratto, l’API deve essere definita esplicitamente. La definizione può avvenire attraverso documentazione (testo, esempi, manuali), oppure attraverso un linguaggio di descrizione standardizzato.

Un linguaggio di descrizione formalizza il contratto, consentendo la generazione automatica di documentazione, codice client e validazione

### OAS - OpenAPI Specification
L’**OAS** (*OpenAPI Specification*) è il linguaggio di descrizione leader del settore per le API moderne basate su HTTP

Si tratta di un formato di descrizione vendor-neutral per le API remote basate su HTTP e rappresenta lo standard industriale per la descrizione delle API moderne, per questo ampiamente adottato dalla comunità di sviluppo

>[!example] Esempio di documento OpenAPI
>I file OpenAPI sono tipicamente scritti in formato YAML (data la sua leggibilità). Questo è un esempio di struttura base di un documento OpenAPI:
>
>```yaml
>openapi: 3.0.0
>info:
>	title: An example OpenAPI document
>	description: |
>		This API allows writing down marks on a Tic Tac Toe 
>		board and requesting the state of the board or
>		of individual cells.
>	version: 0.0.1
>paths: {} # gli endpoint dell'API verrebbero definiti qui
>```

