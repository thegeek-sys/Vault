---
Created: 2025-04-27
Class: "[[Basi di dati]]"
Related:
---
---
## Alfabeto
$$
P=\{\text{persona/1},\text{telefono/2},\text{nome/2}, \text{dipendente/1},\text{dipartimento/1},\text{lavora/2}\}
$$
## Formule

>[!question] Tutte le persone hanno almeno un numero di telefono
>$$\forall x \text{ persona}(x)\to \exists y\text{ telefono}(x,y)$$

>[!question] Ogni persona ha esattamente un nome
>$$\forall x\text{ persona}(x)\to \exists y_{1} \text{ nome}(x,y_{1}) \land \neg\exists y_{2}\Bigl(\text{nome}(x,y_{2})\land\neg(y_{1}=y_{2})\Bigr)$$

>[!question] Non ci sono dipendenti che lavorano in più di due dipartimenti
>$$\forall x\text{ dipendente}(x)\to \forall d_{1},d_{2},d_{3}\Bigl(\bigl(\text{ lavora}(x,d_{1})\land \text{ lavora}(x,d_{2})\land \text{ lavora}(x,d_{3})\bigr)\to(d_{1}=d_{2}\lor d_{1}=d_{3}\lor d_{2}=d_{3})\Bigr)$$

>[!question]