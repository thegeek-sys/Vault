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
Le migliori pratiche per i nomid delle URI