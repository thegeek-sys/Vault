# 1
a)
[[373364331-3b9ed6e2-a3d3-4474-8ebf-1bf0005a79ab.jpg|Esercizio 1 07/11/2018]]
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
\text{VI\_G}=\sigma_{\text{Data}\geq \text{01/01/2018}\land \text{Data}\leq \text{31/01/2018}}(\text{VIAGGIO})
$$
$$
\text{VI\_P}=\pi_{\text{CF, ID, Nome, Cognome, DataNascita}}\left((\text{VI\_G}\underset{\text{ID}=\text{Viaggio}}{\bowtie}\text{EQUIPAGGIO})\underset{\text{Pers}=\text{ID}}{\bowtie}\text{PERSONALE}\right)
$$
$$
\text{PERSONALE}-\text{VI\_P}
$$
