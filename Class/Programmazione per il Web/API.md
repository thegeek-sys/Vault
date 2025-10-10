---
Class: "[[Programmazione per il Web]]"
Related:
---
---
## Introduction
Un’**API** (*Application Programming Interface*) è la definizione delle interazioni consentite tra due parti di un software.

L’API funge da **contratto**, specificando come un pezzo di codice o un servizio può interagire con un altro

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
## Interfaccia e stabilità
La stabilità dell’interfaccia gra