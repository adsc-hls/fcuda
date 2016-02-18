package cetus.hir;

/**
* <b>IDExpression</b> represents an IR object that appears as an identifier in
* the program. In C programs, only the two child classes of IDExpression,
* {@link NameID} and {@link Identifier}, are used.
*/
public abstract class IDExpression extends Expression {

    /** not used in C programs. */
    protected boolean global;

    /** not used in C programs. */
    protected boolean typename;

    /**
    * Creates an IDExpression for derived classes.
    *
    * @param global True if the expression needs to be prefixed by the scope
    *   resolution operator (::), false if it does not need it.
    */
    protected IDExpression(boolean global) {
        super(-1);
        this.global = global;
        this.typename = false;
    }

    /** This constructor is not used in C programs. */
    protected IDExpression(boolean global, boolean typename) {
        super(-1);
        this.global = global;
        this.typename = typename;
    }

    @Override
    public IDExpression clone() {
        IDExpression o = (IDExpression)super.clone();
        o.global = global;
        o.typename = typename;
        return o;
    }

    /**
    * Performs a symbol table lookup of this expression on the nearest
    * enclosing SymbolTable.
    *
    * @return a Declaration for this expression or null if one cannot
    *   be found.
    */
    public Declaration findDeclaration() {
        Traversable t = this;
        if (getParent() == null) {
            System.err.println("expr with no parent");
        }
        while (t != null) {
            if (t instanceof SymbolTable) {
                return ((SymbolTable)t).findSymbol(this);
            } else {
                t = t.getParent();
            }
        }
        return null;
    }

    /** This method is not used in C programs. */
    public void setGlobal(boolean value) {
        global = value;
    }

    /** This method is not used in C programs. */
    public void setTypename(boolean value) {
        typename = value;
    }

    /** Returns name of the IDExpression - this should not be called. */
    public String getName() {
        throw new UnsupportedOperationException();
    }

}
