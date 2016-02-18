package cetus.hir;

/**
* <b>PseudoSymbol</b> is intended to provide <b>Symbol</b> interface in an
* object that is not part of the IR.
*/
public abstract class PseudoSymbol implements Symbol {

    /** Constructs an empty pseudo symbol */
    public PseudoSymbol() {
    }

    /** Returns the string representation of the pseudo symbol */
    @Override
    public String toString() {
        return this.getSymbolName();
    }

    /**
    * This operation is not supported.
    * @throws UnsupportedOperationException always
    */
    public void setName(String name) {
        throw new UnsupportedOperationException();
    }

    /**
    * This operation is not supported.
    * @throws UnsupportedOperationException always
    */
    public Declaration getDeclaration() {
        throw new UnsupportedOperationException();
    }

    /**
    * Returns the most representative symbol object that exist in the IR.
    * e.g., the symbol of the left-hand-side (base address) for accessy symbol.
    * Depending on the sub-class type of <b>PseudoSymbol</b> this method may
    * return null.
    */
    public abstract Symbol getIRSymbol();

    /**
    * This method should be provided for the symbol to be used in collection.
    */
    public abstract boolean equals(Object o);

    /**
    * This method should be provided for the symbol to be used in hashset(map).
    */
    public abstract int hashCode();

}
