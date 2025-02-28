---
Created: 2025-02-28
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
Un algoritmo si dice efficiente se la sua complessità è di ordine polinomiale nella dimensione dell’input.
Ovvero di complessità $O(n^c)$ per una qualche costante $c$

Di conseguenza un algoritmo è inefficiente se la sua complessità è di ordine superpolinomiale:
- **Esponenziale** → è una funzione di ordine $\Theta(c^n)=2^{\theta(n)}$
- **Super-esponenziale** → è una funzione che cresce più velocemente di un esponenziale, ad esempio $2^{\theta(n^2)}$ ma anche $2^{\theta(n\log n)}$
- **Sub-esponenziale** → è una funzione che cresce più lentamente di un esponenziale vale a dire $2^{O(n)}$, ad esempio $n^{\theta(\log n)}=2^{\theta(\log^2 n)}$ oppure $2^{\theta(n^c)}$ dove $c$ è una costante inferiore a $1$

I problemi di cui si conoscono algoritmi subesponenziali e non polinomiali sono pochi e proprio per questo molto studiati. Ad esempio per il problema della fattorizzazione e quello dell'isomorfismo tra grafi sono noti da tempo algoritmi superpolinomiali di complessità $2^{O(n^{1/3})}$

Come caso di studio basti pensare al **test di primalità**.
L'algoritmo banale (dato il numero $n$, cerca un eventuale divisore tra tutti i numeri tra $2$ e $n-1$) è esponenziale perché ha complessità $O(n)$ (nota che la dimensione dell’input è $\log n$).
Un algoritmo efficiente per questo problema dovrebbe avere complessità $O(\log^cn)$ per una qualche costante $c$ mentre questo algoritmo ha complessità $O(2^{\log n})$.
Nella ricerca dell'eventuale divisore fermarsi alla radice velocizza l'algoritmo ma non ne cambia la complessità asintotica che resta esponenziale:
$$
O(\sqrt{ n })=O(2^{\log \sqrt{ n }})=O(2^{\frac{1}{2}\log n})=2^{\theta(\log n)}
$$
Solo nel 2004 si è trovato un algoritmo polinomiale deterministico di tempo $O(\log^{12}n)$ che è stato poi velocizzato a $O(\log^3n)$ anche se con costanti moltiplicative molto alte che non lo rendono competitivo con i ben noti algoritmi probabilistici (degli algoritmi precedenti rendevano il calcolo della primalità possibile in $O(\log^3n)$ ma con un tasso di errore)