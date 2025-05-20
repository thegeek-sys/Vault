---
Created: 2025-05-19
Class: "[[Algoritmi]]"
Related:
---
---
![[Pasted image 20250519162637.png]]

>[!info] Dimostrazione per scambio (swap)
>Supponiamo per assurdo che la strategia greedy **non** produca una soluzione ottima.
>
>Sia:
>- $\text{SOL}$: la soluzione trovata dal greedy.
>- $\text{OPT}$: una soluzione ottima che usa **meno flaconi diversi** da $\text{SOL}$ possibile.
>- $F_1, F_2, \dots$: i flaconi selezionati dal greedy in ordine.
>- Sia $F_i$​: il **primo flacone** scelto da $\text{SOL}$ ma **non** presente in $\text{OPT}$.
>
>Poiché $\text{OPT}$ non contiene $F_i$​, ma entrambi devono contenere almeno PPP pillole, $\text{OPT}$ deve aver usato altri flaconi (più piccoli o meno efficienti) al posto di $F_i$​.
>
>Sia $F' \in \text{OPT}$ un flacone **diverso** da $F_i$​, scelto “al posto” suo, cioè che contribuisce al volume in cui il greedy ha invece inserito le pillole in $F_i$​.

---

## 🔄 Costruzione della nuova soluzione OPT′\text{OPT}'OPT′

Sostituiamo in OPT\text{OPT}OPT il flacone F′F'F′ con FiF_iFi​. Questo è possibile perché:

- Il greedy ha scelto FiF_iFi​ **tra i flaconi rimanenti di capacità massima**.
    
- Quindi FiF_iFi​ ha **capacità maggiore o uguale** rispetto a F′F'F′.
    
- Sostituendo F′F'F′ con FiF_iFi​, otteniamo una nuova soluzione OPT′\text{OPT}'OPT′ con **almeno** la stessa capacità totale di pillole.
    
- Inoltre, OPT′\text{OPT}'OPT′ differisce da SOL\text{SOL}SOL in **meno** flaconi rispetto a OPT\text{OPT}OPT.
    

---

## 🚫 Contraddizione

Abbiamo così costruito una nuova soluzione ottima OPT′\text{OPT}'OPT′, che:

- È **ancora ottima**, perché contiene abbastanza pillole.
    
- **Differisce meno** da SOL\text{SOL}SOL rispetto a OPT\text{OPT}OPT, **contraddicendo** la scelta di OPT\text{OPT}OPT come quella più vicina a SOL\text{SOL}SOL.
    

---

## ✅ Conclusione

L’ipotesi che SOL\text{SOL}SOL **non sia ottima** porta a una contraddizione.  
Quindi la strategia greedy che seleziona **il flacone con capacità massima residua** ad ogni passo è **ottima per minimizzare il numero di flaconi**.



