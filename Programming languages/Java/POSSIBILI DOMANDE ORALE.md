## slide 2/4 (oggetti)

>[!Question]- incapsulamento: cos'è e perché farlo
>- l'**incapsulamento** è il processo che nasconde i dettagli realizzativi (campi e implementazione), rendendo pubblica un’interfaccia (metodi pubblici).
>- aiuta a *modularizzare* lo sviluppo, creando un funzionamento "a scatola nera"
>- non è sempre necessario sapere tutto
>- facilita la maintenance

>[!Question]- parola chiave "static" (contributo slide più avanti: classi statiche)
>campi
>- un campo statico è relativo all'**intera classe** e non a un'istanza.
>- esiste in una sola locazione di memoria (nel MetaSpace)
> 
>metodi
>- si può accedere a un metodo statico da dentro una classe con la sua segnatura, o da fuori con la segnatura `Classe.metodo()`
>- non si possono usare i campi di classe ma solo statici
>
>classi
>- una classe interna static non richiede l'esistenza di un oggetto della classe che la contiene e non ha riferimento implicito ad essa
>- non ha accesso allo stato degli oggetti della classe che la contiene

>[!Question]- enumerazioni
>- sono dei tipi i cui valori possono essere scelti tra un **insieme predefinito** di identificatori univoci.
>- le costanti enumerative sono implicitamente static
>- non è possibile creare un oggetto di tipo enum
>- il compilatore genera un metodo statico `values` che restituisce un array delle enum

>[!Question]- classi wrapper
>- permettono di convertire i valori dei primitivi in oggetti
>- sfruttano i meccanismi di **auto-boxing** (conversione automatica di un primitivo al suo wrapper) e **auto-unboxing** (l'inverso)

## slide 5/6 (ereditarietà e polimorfismo)

>[!Question]- classi e metodi astratti
>- una classe astratta non può essere istanziata, ma verrà estesa da classi che possono essere istanziate
>- un metodo astratto (definibile esclusivamente in una classe astratta) non viene implementato (ma solo definito con la sintassi `abstract tipo nome();`) - **tutte le classi non astratte che estendono una classe astratta devono implementare i suoi metodi astratti**

>[!Question]- this e super nei costruttori
>entrambi vanno collocati obbligatoriamente nella prima riga del costruttore.
>- `this` permette di chiamare un altro costruttore della stessa classe
>- `super` permette di chiamare il costruttore della superclasse
>se la superclasse non fornisce un costruttore senza argomenti, la sottoclasse deve esplicitamente definire un costruttore (infatti una sottoclasse deve necessariamente chiamare il costruttore della sua superclasse)

>[!Question]- differenza tra overriding e overloading
>- l'overriding è una ridefinizione (reimplementazione) di un metodo c**on la stessa segnatura** di una sopraclasse
>	- gli argomenti devono essere gli stessi
>	- il tipo di ritorno deve essere compatibile (lo stesso o una sua sottoclasse)
>	- non si può ridurre la visibilità
>- l'overloading è la creazione di un metodo con lo stesso nome ma con una **segnatura alternativa**
>	- i tipi di ritorno possono essere diversi (ma questo non può essere l'unico cambiamento)
>	- si può variare la visibilità a piacimento

>[!Question]- tipi di visibilità
>- **private** - visibile solo all'interno della classe
>- **public** - visibile a tutti
>- **default/package** - visibile all'interno del pacchetto (il costruttore delle enum lo ha di default)
>- **protected** - visibile all'interno del pacchetto e alle sottoclassi

>[!Question]- is-a vs has-a
>- is-a rappresenta l'**ereditarietà**
>- has-a rappresenta la **composizione** - un oggetto contiene come membri riferimenti ad altri oggetti

>[!Question]- cos'è il polimorfismo?
>- il polimorfismo permette a una variabile di un certo tipo di contenere un riferimento a un oggetto di qualsiasi sua sottoclasse.
>- vengono chiamati i metodi in base al tipo effettivo dell'oggetto
>- il polimorfismo sfrutta il **binding dinamico** - l'associazione tra una variabile e un metodo viene stabilita a runtime

>[!Question]- come funzionano le conversioni con il polimorfismo?
>- si può creare un oggetto di una sottoclasse e assegnarlo a una variabile della superclasse
>- per fare `downcasting`, invece, serve casting esplicito
>- quando si fa `upcasting`, si possono chiamare solo i metodi e vedere solo i campi della superclasse

>[!Question]- classi e metodi final
>- la parola chiave final impedisce di creare sottoclassi o di reimplementare metodi

## slide 7 (interfacce)

>[!Question]- cos'è un'interfaccia?
>- un'interfaccia specifica il **comportamento** che un oggetto deve presentare all'esterno - l'implementazione delle operazioni non viene definita
>- un'interfaccia è una classe astratta al 100%

>[!Question]- caratteristiche (componenti) delle interfacce
>- è possibile definire implementazione di metodi statici o di default all'interno di un'interfaccia (i metodi statici non godono di polimorfismo vs google : Default methods allow you to add methods to existing interfaces without breaking existing implementations.)
>- tutti i metodi di un'interfaccia sono implicitamente `public abstract`
>- tutti i campi di un'interfaccia sono implicitamente `public static final`
>- in Java è permessa l'implementazione di molteplici interfacce, mentre non è permessa l'ereditarietà multipla

>[!Question]- iterable e iterator
>- iterator è un'interfaccia che permette di iterare su collezioni. espone i metodi `hasNext()`, `next()` e `remove()`
>- iterator è in relazione con l'interfaccia `Iterable` - chi implementa `Iterable` restituisce un `Iterator`

>[!Question]- classi nested e inner
>- le classi presenti all'interno di altre classi si chiamano **nested classes**. queste si definiscono **inner** se non sono statiche.
>- per istanziare una classe inner, è necessario prima istanziare la classe esterna che la contiene. (ogni classe interna ha un riferimento implicito alla classe che la contiene). dalla classe interna si può accedere a tutte le variabili e a tutti i metodi della classe esterna.
>- una classe annidata statica non richiede l'esistenza di un oggetto della classe esterna, e non ha riferimenti impliciti ad essa.

>[!Question]- interfacce SAM
>- un'interfaccia SAM è un'interfaccia funzionale composta da un singolo metodo astratto (single abstract method)
>- a ogni metodo che accetta una SAM si può passare una lambda compatibile

>[!Question]- classi anonime vs espressioni lambda
>- `this` - nelle classi anonime si riferisce all'oggetto anonimo, nelle lambda all'oggetto che le contiene
>- compilazione - le anonime sono compilate come classi interne, le lambda come metodi privati

## slide 8 (strutture dati)

>[!Question]- gerarchia degli iterabili (collections)
>![[gerarchia collections.png|center|300]]
> - `Collection` è un'interfaccia alla base della gerarchia delle collezioni
> - `Set` è una collezione senza duplicati
> - `List` è una collezione ordinata che può contenere duplicati
> - `Map` associa coppie chiave-valore senza chiavi duplicate
> - `Queue` è una collezione FIFO (coda)
>   
>  altre sottocollezioni fondamentali: `ArrayList, LinkedList, HashSet, TreeSet, HashMap, TreeMap (ordinamento sulle chiavi)...`

>[!Question]- come si può iterare sulle collezioni?
>- con gli iterator
>- con un foreach
>- mediante indici (solo per le liste)

>[!Question]- riferimenti a metodi
>- sintassi `Classe::metodoStatico` oppure `oggetto::metodoNonStatico` oppure `classe::metodoNonStatico` - la differenza tra il riferimento alla classe e all'oggetto è che, nel caso della classe, non stiamo specificando su che oggetto applicare il metodo

>[!Question]- a cosa possono accedere le lambda?
>- (simile alle anonime) ai campi di istanza e variabili statiche
>- a variabili final del metodo che le definisce

>[!Question]- tipi di SAM
>- `Predicate<T>` funzione booleana a un solo argomento generico
>- `Function<T, R>` - funzione a un argomento e un tipo di ritorno generici
>- `Supplier<T>` - funzione con un argomento generico e nessun tipo di ritorno

## slide 9 (tipi generici)

>[!Question]- sintassi tipi generici
>- nelle classi, vanno inseriti dopo il nome della classe `public class Class<T>`
>- nei metodi, prima del tipo di ritorno `public static <T> void`

>[!Question]- come funzionano i generici negli array?
>- è possibile creare un array di tipo `T extends Classe`, ma questo rende essenzialmente impossibile l'aggiunta di elementi all'array (visto che non si può avere un array eterogeneo)
>- si può quindi creare un array del supertipo e aggiungere ad esso gli elementi dei diversi sottotipi - java farà upcasting a tempo di compilazione

>[!Question]- il jolly
>- nel caso in cui non sia necessario usare il tipo generico nel corpo di una classe che necessita di generici, è possibile utilizzare il jolly `?` (wildcard)

>[!Question]- come funziona la cancellazione del tipo?
>quando si utilizza un tipo generico nella segnatura di un metodo/classe, il compilatore, nel tradurlo in bytecode, elimina la sezione del tipo parametrico e lo sostituisce con quello **reale**. (per esempio, di default, T viene sostituito con Object)

>[!Question]- come funziona PECS?
>"Producer Extends, Consumer Supers"
>- `extends` e `super` esistono per due necessità principali: leggere e scrivere in una collezione generica.
> 
>ci sono 3 modi per farlo:
>- `<?>` - non so nulla sul tipo, posso solo leggere ma non scrivere
>- `extends` - so qualcosa sul tipo, posso comunque solo leggere ma posso svolgere operazioni sugli elementi della collezione (es. se sono `? extends Integer` usare le operazioni di Integer). infatti, il tipo dato dal `get` è quello che viene esteso
>- `super` - posso leggere e scrivere nella lista (e posso metterci super e sub classi del tipo specificato), ma non posso assumere il tipo (il `get` mi ritornerà un Object)

>[!Question]- come ottenere informazioni sull'istanza di un generico
>per via della cancellazione del tipo generico, non si può conoscere il tipo generico a tempo di esecuzione. per controllare il tipo di una collection, quindi, `instanceof List<Integer>` non funziona, mentre `instanceof List<?>` sì.
>lista t e lista ?

>[!Question]- vincoli sul tipo generico
>`super` → il tipo generico deve essere una superclasse della classe specificata o la classe stessa (controvarianza)
>`extends` → deve necessariamente essere un sottotipo di `Classe` (o la classe stessa) o implementare `Interfaccia` (covarianza)

>[!Question]- overloading di metodi con tipo generico
>un metodo generico piò essere sovraccaricato come ogni altro metodo e inoltre da un metodo non generico con stesso nome e numero di parametri
>viene sempre prima cercato un metodo specifico e in caso questo non sia presente viene eseguito il metodo generico

>[!Question]- come funziona la cancellazione del tipo?
>>Infatti quando il compilatore traduce il metodo/la classe generica in bytecode Java:
>1. **elimina la sezione del tipo parametrico** e sostituisce il tipo parametrico con quello reale
>2. per default **il tipi generico viene sostituito** con il tipo `Object` (a meno di vincoli sul tipo)

## slide 10 (eccezioni)

>[!Question]- throware e catchare le eccezioni

>[!Question]- differenza errori ed exception (gerarchie exception)

>[!Question]- blocco finally
>viene eseguito indipendentemente se viene throwato un errore oppure no
>in caso di un altra istruzione 

>[!Question]- perché dovresti catchare errori più specifici?

>[!Question]- come fare eccezioni personalizzate?

>[!Question]- eccezioni checked e unchecked
>java ti obbliga o no


## slide 11/12 (file, ricorsione, stream)
>[!Question]- come funziona la mutua ricorsione?
>Si parla di muta ricorsione nel caso in cui si hanno due o più funzioni che singolarmente non sono ricorsive ma si chiamano vicendevolmente

>[!Question]- in che modi si può leggere un file?
>bufferedreader reader scanner printwriter formatter

>[!Question]- optional
>è un contenitore (interfaccia) di un riferimento che potrebbe essere o non essere `null` (in modo tale che un metodo può restituire un `Optional` invece di restituire  un riferimento potenzialmente `null`, per  evitare i `NullPointerException`)

