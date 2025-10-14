---
Class: "[[Programmazione per il Web]]"
Related:
---
---
## Introduction
Il **REST** (*Representational State Transfer*) è uno stile architetturale per sistemi impermediali distribuito. E’ stato proposto per la prima volta da Roy Fielding nel 2000 e il suo obiettivo è trasferire la rappresentazione delle risorse da un componente (es. server) ad un altro (es. client)

Vediamo però ora qualche termine

>[!info] Resource
>Una risorsa è una qualsiasi informazione che possa essere nominata (es. documento, immagine, servizio, collezione di altre risorse) e che può variare nel tempo
>
>Due risorse però possono mappare agli stessi valori in un dato momento (es. “versione X.X” di un programma e “ultima versione”)

>[!info] Representation
>La **rappresentazione della risorsa** è lo stato attuale o previsto di una risorsa, ovvero il valore della risorsa in un momento particolare ed è composta da *data e metadata* (il formato dei dati è noto come *media type*)
>
>I componenti REST (client o server) eseguono azioni su una risorsa usando una rappresentazione

>[!info] Resource Identifiers
>Gli identificatori sono usati per **identificare** (indirizzare) una risorsa. Una **Uniform Resource Identifier** (*URI*) è una sequenza unica di caratteri che identifica una risorsa logica o fisica

---
## URI
Una **Uniform Resource Identifier** (*URI*) è una sequenza unica di caratteri che identifica una risorsa logica o fisica

### Best practices
Le migliori pratiche per i nomi delle URI:
- utilizzare *sostantivi* per rappresentare le risorse
- utilizzare *sostantivi singolari* per una singola risorsa
- utilizzare *sostantivi plurali* per una collezione di risorse

### Notation
Convenzioni di notazione per le URI:
- forward slash (`/`) per esprimere gerarchia
- preferire i trattini (`-`) agli underscore (`_`)
- utilizzare solo *lettere minuscole*
- non utilizzare *estensioni di file* (il media type è comunicato negli header)
- utilizzare la *componente query* per filtrare

>[!tip]
>Usare il trailing slash solo se la risorsa non è una foglia

### Principi di formulazione URI

| Regola                      | Esempio positivo     | Esempio negativo                   |
| --------------------------- | -------------------- | ---------------------------------- |
| Risorsa singola (singolare) | `/users/45`          | `/get-user/45` (contiene un verbo) |
| Collezione (plurale)        | `/invoices`          | `/invoice-list`                    |
| Gerarchia                   | `/users/45/orders`   | `/orders-from-user/45`             |
| Separazione                 | preferire i `-`      | evitare gli `_`                    |
| Media type                  | non usare estensioni | `/products/123.json`               |

>[!warning] Operations: not in the URI
>Le URI sono usate per identificare in modo univoco le **risorse** e non le azioni su di esse. Azioni diverse possono essere eseguite su una risorsa attraverso i metodi supportati (l’interfaccia tra i componenti)
>
>>[!example] Esempi di metodi sulla stessa URI
>>`GET`, `PUT` e `DELETE` sull’URI `http://example.com/managed-devices/{id}`

---
## RESTful system constraints
I vincoli di un sistema RESTful sono:
1. client-server
2. stateless
3. cacheable
4. uniform-interface
5. layered system

### Client-server
Applica la **separazione delle responsabilità** (il client gestisce l’UI, il server gestisce l’archiviazione dei dati) e migliora la portabilità e scalabilità

### Stateless
Ogni **richiesta del client deve contenere tutte le informazioni** necessarie per essere compresa e non può dunque sfruttare alcun contesto memorizzato sul server

Lo stato della sessione è mantenuto internamente sul client, mentre lo stato della risorsa è mantenuto sul server

### Cacheable
Il client può in seguito **riutilizzare una rappresentazione** della risorsa (dati) che è considerata *cacheable*. Il periodo di tempo per cui la risorsa può essere messa in cache è specificato nella risposta

### Uniform interface
Un’interfaccia uniforme promuove la **standardizzazione** (a discapito dell’efficienza) e le implementazioni sono disaccoppiate dai servizi che forniscono

I quattro vincoli dell’interfaccia sono:
- identificazione delle risorse
- manipolazione delle risorse tramite rappresentazioni
- messaggi auto-descrittivi
- hypermedia come motore dello stato dell’applicazione (il client necessita solo della URI iniziale)

>[!example]
>API REST basate su HTTP usano metodi standard (GET, POST, PUT, …)


### Layered system
Nella comunicazione possono essere coinvolti diversi componenti in un’architettura a strati. I componenti intermedi agiscono sia come client che come server; inoltrano richieste e risposte, a volte con una traduzione

Inoltre ogni componente non può “vedere” oltre lo stato immediatamente adiacente con cui interagisce

