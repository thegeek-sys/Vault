# [[373364331-3b9ed6e2-a3d3-4474-8ebf-1bf0005a79ab.jpg|Esercizio 1 07/11/2018]]
a)
$$
\text{V\_2018}=\sigma_{\text{Data}\geq \text{01/01/2018 } \land \text{ Data}\leq \text{31/12/2018}}(\text{VIAGGIO})
$$
$$
\text{A\_NOR}=\sigma_{\text{AnnoR}=\text{00/00/00}}(\text{AEREO})
$$
$$
\text{VOLO\_R}=\sigma_{\text{Città="Roma"}}(\rho_{\text{VSigla}\leftarrow \text{Sigla}}(\text{VOLO})\underset{\text{Arrivo}=\text{Sigla}}{\bowtie}\text{AEROPORTO})
$$
$$
\pi_{\text{Data}}(\text{VOLO\_R}\underset{\text{VOLO\_R.Sigla}=\text{A\_2018.SiglaVolo}}{\bowtie} \text{A\_2018}\underset{\text{Aereo}=\text{A\_NOR.ID}}{\bowtie} \text{VOLO\_NOR})
$$
b)
$$
\text{G18}=\rho_{\text{IDV}\leftarrow\text{ID}}\left(\sigma_{\text{Data}\geq \text{01/01/2018}\land \text{Data}\leq \text{31/01/2018}}(\text{VIAGGIO})\right)
$$
$$
\text{VI\_P}=\pi_{\text{CF, ID, Nome, Cognome, DataNascita}}\left((\text{G18}\underset{\text{IDV}=\text{Viaggio}}{\bowtie}\text{EQUIPAGGIO})\underset{\text{Pers}=\text{ID}}{\bowtie}\text{PERSONALE}\right)
$$
$$
\text{PERSONALE}-\text{VI\_P}
$$
# [[Pasted image 20241103155926.png|Esercizio 1 11/02/2016]]
a)
$$
\text{AD}=\sigma_{\text{Data=25-01-2026}}(\text{NOLEGGIO})
$$
$$
\text{AM}=(\sigma_{\text{Tipo=persone}}(\text{AUTOVEICOLO})) \underset{\text{Modello=Nome}}{\bowtie}\text{MODELLO}
$$
$$
\pi_{\text{Targa, Posti}}(\text{AM}\bowtie\text{AD})
$$

b)

$$
\text{MD}=\sigma_{\text{Motore=disel}}(\text{MODELLO})
$$
$$
\text{r}=(\text{MD}\underset{\text{Nome=Modello}}{\bowtie}\text{AUTOVEICOLO})\bowtie\sigma_{\text{Data}\geq \text{01/01/2016}\land \text{Data}\leq \text{31/01/2016}}(\text{NOLEGGIO})
$$
$$
\sigma_{\text{Targa}}(\text{MD}\bowtie\text{AUTOVEICOLO}-\pi_{\text{Targa}}(\text{r}))
$$

# [[Pasted image 20241117172103.png|Esercizio 2 11/02/2016]]
a)
In base a quanto visto basta verificare che $F\subseteq G^+$ cioè che ogni dipendenza funzionale in $F$ si trova in $G^+$

>[!info]
>Come abbiamo verificato è inutile controllare che vengano preservate le dipendenze tali che l’unione delle parti destra e sinistra è contenuta interamente in un sottoschema, perché secondo la definizione $\pi_{R_{i}}(F)=\{X\to Y \mid X\to Y\in F^+\land XY\subseteq R_{j}\}$
>
>In questo esempio vale per $A\to BD$, per $B\to E$ e per $D\to C$
>Quindi verifichiamo solo che venga preservata la dipendenza $CE\to A$

Verifichiamo che sia preservata $CE\to A$
$Z=CE$
$S=\varnothing$
Ciclo esterno sui sottoschemi $ABD$ e $BCDE$
$$S=S\cup(CE\cap ABD)^+_{F}\cap ABD=\varnothing\cup(\varnothing)^+_{F}\cap ABD=\varnothing$$
$$\begin{align}
S&=\varnothing\cup(CE\cap BCDE)^+_{F}\cap BCDE=\varnothing\cup(CE)^+_{F}\cap BCDE=\\&=\varnothing\cup ABCDE\cap BCDE=BCDE
\end{align}$$

$BCDE\not\subset CE$ quindi entriamo nel ciclo while
$Z=Z\cup S=BE\cup BCDE=BCDE$
Ciclo for interno al while sui sottoschemi $ABD$ e $BCDE$
$$\begin{align}
S&=BCDE\cup(BCDE\cap ABD)^+_{F}\cap ABD=BCDE\cup(BD)^+_{F}\cap ABD=\\&=BCDE\cup ABCDE\cap ABD=ABCDE
\end{align}
$$

A questo punto ci possiamo fermare, perché a $S$ vengono sempre e solo aggiunti elementi, e ora $S$ contiene tutto $R$, e in particolare contiene $A$, quindi $A\subseteq(CE)^+_{G}$ e quindi $CE\to A\in G^A=G^+$
In conclusione possiamo dire che la decomposizione data preserva $F$
