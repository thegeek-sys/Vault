Un oggetto ha:
- **stato**
- operazioni
per effettuare una richiesta a un oggetto, si chiama un metodo di quell'oggetto (funzione).

Ogni oggetto **incapsula** altri oggetti (es. macchinetta di caffè ha acqua, caffè...), e un nuovo tipo di oggetto può essere creato utilizzandone altri già esistenti.
L'incapsulamento viene utilizzato anche per l'information hiding - celare alcuni procedimenti/informazioni che non devono essere visibili all'utente.

Ogni oggetto ha un suo tipo - una **classe**, di cui è un'**istanza**.
La classe definisce il comportamento di un oggetto, attraverso una serie di **metodi**.

**EREDITARIETÀ** - non serve ricreare nuove classi di oggetti se queste hanno funzionalità simili.
Si definisce una superclasse, da cui derivano una serie di classi diverse (es. superclasse "forma", classi "cerchio", "quadrato"...).

**POLIMORFISMO** - è possibile utilizzare la classe base senza dover conoscere la classe specifica di un oggetto, e quindi scrivere codice che <u>non dipende dalla classe specifica</u> (es. non è necessario sapere che un cerchio è un cerchio, basta sapere che è un sottotipo di forma - basta lavorare sulla classe forma -).