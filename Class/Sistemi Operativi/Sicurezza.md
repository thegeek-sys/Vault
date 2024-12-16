---
Created: 2024-12-16
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
Iniziamo riportando la definizione di **sicurezza informatica** del NIST (National Institute of Standards and Technology) ovvero: “è la protezione offerta da un sistema informativo automatico al fine di conservare integrità, disponibilità e confidenzialità delle risorse del sistema stesso”

### La triade
![[Pasted image 20241216115902.png|center]]

Dunque ci sono tre obiettivi che costituiscono il cuore della sicurezza:
- **integrità**
- **disponibilità**
- **confidenzialità**

Ci sono due ulteriori obiettivi che vengono aggiunti al nucleo della sicurezza informatica:
- autenticità
- tracciabilità

---
## Gli obiettivi nel dettaglio
Analizziamo ora i tre obiettivi più nel dettaglio:
- **Integrità** → riferita tipicamente ai dati, che non devono essere modificati senza le dovute autorizzazioni
- **Confidenzialità** → riferita tipicamente ai dati, che non devono essere letti senza le dovute autorizzazioni
- **Disponibilità** → riferita tipicamente ai servizi, che devono essere disponibili senza interruzioni
- **Autenticità** → riferita tipicamente agli utenti ,che devono essere chi dichiarano di essere (per estensione vale anche per messaggi e dati)

---
## Minacce (threats)
L’**RFC 2828** descrive quattro conseguente delle minacce informatiche
- accesso non autorizzato (*unauthorized disclosure*)
- imbroglio (*deception*)
- interruzione (*disruption*)
- usurpazione (*usurpation*)

### Accesso non autorizzato
Si verifica un accesso non autorizzato quando un’entità ottiene l’accesso a dati per i quali non ha autorizzazione. Ciò costituisce una minaccia alla confidenzialità

Tipicamente gli attacchi ad un SO che riescono ad ottenere un accesso non autorizzato sono:
- esposizione (intenzionale o per errore) → ciò che dovrebbe essere privato è invece pubblico
- intercettazione → attaccante che si mette in mezzo ad una comunicazione
- inferenza → riesco a dedurre alcuni dati dai dati pubblici
- intrusione → attaccante riesce ad entrare direttamente in un sistema

### Imbroglio
Avviene un imbroglio quando un’entità autorizzata riceve dati falsi e pensa che siano veri. Ciò costituisce una minaccia all’integrità

Questo tipo di minaccia può avvenire per:
- mascheramento → l’attaccante riesce ad entrare in possesso delle credenziali di un utente autorizzato (trojan)
- falsificazione (es. uno studente che modifica i propri voti)
- ripudio → quando un utente nega di aver ricevuto o inviato dati

### Interruzione
L’interruzione consiste nell’impedimento al corretto funzionamento dei servizi, e costituisce una minaccia all’integrità del sistema o alla disponibilità

Questo tipo di minaccia può avvenire per:
- incapacitazione → rompendo qualche componente del sistema
- ostruzione → Denial of Service (DoS), per esempio riempiendo il sistema di richieste
- corruzione → alterazione dei servizi

### Usurpazione
Si parla di usurpazione quando il sistema viene direttamente controllato da chi non ne ha l’autorizzazione