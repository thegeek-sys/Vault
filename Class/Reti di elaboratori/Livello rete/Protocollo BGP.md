---
Created: 2025-05-07
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Struttura di internet
![[Pasted image 20250507215644.png|center|500]]

Gli ISP forniscono servizi a livelli differenti:
- **dorsali** → gestite da società private di telecomunicazioni, che forniscono la connettività globale (connesse tramite peering point)
- **network provider** → che utilizzano le dorsali per avere connettività globale e forniscono connettività ai clienti Internet
- **customer network** → che usano i servizi  dei network provider

### Impossibilità di usare un singolo protocollo di routing
Abbiamo fin qui visto la rete come una collezione di router interconnessi in cui ogni router era indistinguibile dagli altri

Però nella pratica si hanno 200 milioni di destinazioni e archiviare le informazioni d’instradamento su ciascun host richiederebbe un’enorme quantità di memoria, il traffico generato dagli aggiornamenti link state non lascerebbero banda per i pacchetti di dati e il distance vector non convergerebbe mai

Per questi motivi 