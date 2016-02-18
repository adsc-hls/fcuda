package cetus.hir;

/** Represents an immediate value within the program. */
public abstract class Literal extends Expression {

    /** Constructs an empty literal */
    protected Literal() {
        // Literals are always leaves, so we can save
        // a lot of memory by not making a child list.
        super(-1);
    }
    
    @Override
    public Literal clone() {
        Literal o = (Literal)super.clone();
        return o;
    }

}
