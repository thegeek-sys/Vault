---
Created: 2024-05-05
Class: "[[Architettura degli elaboratori]]"
Related:
  - "[[Pipeline]]"
Completed:
---
---
## Introduction
Adesso che è chiaro cosa sia una pipeline possiamo iniziare a pensare a come implementarla all’interno di una CPU MIPS.
Per farlo e permettere quindi il **forwarding** (operazione cardine all’interno della pipeline) è necessario integrare dei registri tra le varie unità funzionali, in modo tale da poter recuperare un dato se necessario
![[registri pipeline_1.jpeg]]


