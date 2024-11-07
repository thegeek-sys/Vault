---
Created: 2024-11-07
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Chiavi di uno schema di relazione
Utilizziamo il calcolo della chiusura di un insieme di attributi per determinare le chiavi di uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$

>[!example]
>$$R=(A,B,C,D,E,H)$$
>$$F=\{AB\to CD,C\to E,AB\to E, ABC\to D\}$$
>
>Calcolare la chiusura dell’insieme $ABH$
>
>$$
\begin{flalign}
&\text{begin}\\
&Z:=\color{red}ABH\\
&S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}=\{C \text{ (per la dipendenza }\textcolor{red}{AB}\to\textcolor{royalblue}{CD}\text{)},\\& E\text{ (per la dipendenza }\textcolor{red}{AB}\to\textcolor{royalblue}{CD}\text{)}\}\\
&\text{while } S\not\subset Z\\
&\qquad\text{do}\\
&\qquad\text{begin}\\
&\qquad\qquad Z:=Z\cup S\\
&\qquad\qquad S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}\\
&\qquad\text{end}\\
&\text{end}\\
\end{flalign}
$$
