---
Created: 2024-11-21
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
fino ad ora abbiamo parlato del perché possa essere necessario decomporre uno schema di relazione $R$, su cui è definito un insieme di dipendenze funzionali $F$, soprattutto in relazione a violazioni della 3NF che causano diversi tipi di anomalie

Abbiamo detto più volte che, qualunque sia il motivo che ci porta a decomporre lo schema, la decomposizione deve soddisfare tre requisiti fondamentali:
- ogni sottoschema deve essere 3NF
- la decomposizione deve preservare le dipendenze funzionali
- deve essere possibile ricostruire ogni istanza legale dello schema originale tramite join naturale di istanze della decomposizione

Nelle lezioni precedenti abbiamo visto come verificare che una decomposizione data (non ci interessa come sia sta prodotta) soddisfi tutte le indicazioni, in particolare abbiamo parlato di come verificare:
- se la decomposizione preserva le dipendenze funzionali ([[Decomposizioni che preservano le dipendenze|qui]])
- se sarà possibile ricostruire ogni istanza legale dello schema originale tramite join naturale di istanze della decomposizione ([[Decomposizioni che hanno un join senza perdita|qui]])

### Che si fa ora?
Ora affrontiamo il problema di come ottenere una decomposizione che soddisfi le nostre condizioni.
Prima di tutto: è sempre possibile ottenerla? Si, è sempre possibile, dato uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$, decomporlo in modo da ottenere che:
- ogni sottoschema è 3NF
- la decomposizione preserva le dipendenze funzionali
- è possibile ricostruire ogni istanza legale dello schema originale tramite join naturale di istanze della decomposizione

Presenteremo un algoritmo che raggiunge questo scopo

>[!warning]
>Per la 3NF è sempre possibile applicare l’algoritmo di decomposizione (dopo aver trovato la copertura minimale), per la Boyce-Codd non esiste

---