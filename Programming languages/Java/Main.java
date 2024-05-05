package PECS;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class Main {

    public static <T> void copy(T[] src, T[] dst) {
        dst[0] = src[0];
        // src producer
        // dst consumer (consuma ciò che viene prodotto)
    }

    public static <T> void copyListNOPECS(List<T> src, List<T> dst) {
        for (T o : src)
            dst.add(o);
    }

    public static <T> void copyListPECS(List<? extends T> src, List<? super T> dst) {
        for (T o : src)
            dst.add(o);
    }



    public static void main(String[] args) {

        // il PECS in questo caso non ci può risolvere questo problema in quanto è applicabile solo alle Collection
        // utilizzo le Collection poiché i tipi delle Collection sono generici (non posso mettere un tipo generico in un array)
        Mela[] mele = new Mela[] {new Mela(), new Mela()};
        Pera[] pere = new Pera[] {new Pera(), new Pera()};

        // polimorfismo agisce sugli array in java, sia array di Mela che array di Pera sono array di Frutto
        // non ho errori sull'ide perché l'errore di consistenza sul tipo vengono fatti solamente dal compilatore
        // il compilatore mi restituisce ArrayStoreException
        copy(mele, pere);


        // provo a fare la copia di arraylist di tipi diversi senza utilizzare PECS
        List<Mela> lmele = Arrays.asList(mele); // mi darà errore, sono liste immutabili
        List<Pera> lpere = Arrays.asList(pere);
        copyListNOPECS(lmele, lpere); // NON rispetta il vincolo di essere dello stesso tipo T
        copyListNOPECS(lmele, lmele); // rispetta il vincolo di essere dello stesso tipo T

        // provo quindi a creare un arraylist di frutti ma anche in questo caso mi viene dato errore
        List<Frutto> lfrutti = Arrays.asList(new Frutto[]{new Pera(), new Mela()});
        copyListNOPECS(lmele, lfrutti); // List<Mela> non può essere convertito a List<Frutto>, non vale il polimorfismo

        copyListPECS(lmele, lfrutti); // il tipo della lista di destinazione può essere Mela, Frutto o Object

        Set<Frutto> slf = new TreeSet<>((a, b) -> a.compareTo(b));

    }
}
