package cetus.hir;

import java.util.LinkedList;
import java.util.List;

/**
* Class DerefSymbol represents symbol object that does not exist in the
* IR as a variable declarator but is accessible through pointer dereferencing.
*/
public class DerefSymbol extends PseudoSymbol implements Symbol {

    // The reference symbol which this deref symbol is dereferenced from.
    private Symbol ref_symbol;

    /**
    * Checks if the given symbol produces an deref symbol and returns a
    * single-depth deref symbol if it does.
    * @param symbol the given symbol object.
    * @return a new deref symbol if possible, null otherwise.
    */
    @SuppressWarnings("unchecked")
    public static Symbol get(Symbol symbol) {
        List specs = symbol.getTypeSpecifiers();
        if (isPointerParameter(symbol)) {
             specs.add(PointerSpecifier.UNQUALIFIED);
        }
        if (specs.get(specs.size() - 1) instanceof PointerSpecifier) {
             return new DerefSymbol(symbol);
        } else {
             return null;
        }
    }

    /**
    * Constructs a new deref symbol from the given reference symbol object.
    * @param symbol the reference symbol.
    */
    public DerefSymbol(Symbol symbol) {
        ref_symbol = symbol;
    }

    // Checks if the given symbol is pointer formal parameter without pointer
    // specifier.
    private static boolean isPointerParameter(Symbol symbol) {
        return (!(symbol instanceof PseudoSymbol) &&
                SymbolTools.isFormal(symbol) &&
                !symbol.getArraySpecifiers().isEmpty());
    }

    /**
    * Returns the list of specifiers for this deref symbol.
    * @return the list of specifiers.
    */
    @SuppressWarnings("unchecked")
    public List getTypeSpecifiers() {
        LinkedList ret = new LinkedList();
        for (Object spec : ref_symbol.getTypeSpecifiers()) {
            if (!spec.equals(Specifier.EXTERN)) {
                ret.add(spec);
            }
        }
        // Intentionally adds a pointer specifier for formal parameters having
        // array specifier but not having pointer specifier.
        if (isPointerParameter(ref_symbol)) {
            ret.add(PointerSpecifier.UNQUALIFIED);
        }
        if (!(ret.removeLast()instanceof PointerSpecifier)) {
            throw new InternalError("Invalid DerefSymbol object: " + this);
        }
        return ret;
    }

    /**
    * Returns the list of array specifiers of the reference symbol.
    * @return the list of array specifiers.
    */
    public List getArraySpecifiers() {
        return ref_symbol.getArraySpecifiers();
    }

    /**
    * Returns the name of this deref symbol.
    * @return the name string for this deref symbol.
    */
    public String getSymbolName() {
        return ("*" + ref_symbol.getSymbolName());
    }

    /**
    * Returns the reference symbol (e.g., ref_symbol == &this_deref).
    * @return the reference symbol to this deref symbol.
    */
    public Symbol getRefSymbol() {
        return ref_symbol;
    }

    /**
    * Returns the IR symbol that represents this deref symbol. It returns,
    * basically, the reference symbol, but may return the IR symbol by
    * recursively searching for the IR symbol if necessary (e.g., if the
    * reference symbol is not an IR symbol).
    * @return the representative IR symbol.
    */
    public Symbol getIRSymbol() {
        if (ref_symbol instanceof PseudoSymbol) {
            return ((PseudoSymbol)ref_symbol).getIRSymbol();
        } else {
            return ref_symbol;
        }
    }

    /**
    * Returns the depth of dereferencing.
    * @return the depth.
    */
    public int getDepth() {
        int ret = 1;
        Symbol ref = ref_symbol;
        while (ref instanceof DerefSymbol) {
            ref = ((DerefSymbol)ref).getRefSymbol();
            ret++;
        }
        return ret;
    }

    /**
    * Returns the visible symbol corresponding to this deref symbol.
    * getIRSymbol provides similar functionality but provides base symbol in
    * case of StructSymbol. See SymbolTools
    * @return visible symbol
    */
    public Symbol getVisibleSymbol() {
        Symbol ret = null;
        ret = ref_symbol;
        while (ret instanceof DerefSymbol) {
            ret = ((DerefSymbol)ret).getRefSymbol();
        }
        return ret;
    }

    /**
    * Returns an expression instantiation of this deref symbol.
    * @return a new instance of expression from this deref symbol.
    */
    public Expression toExpression() {
        Expression ret;
        if (ref_symbol instanceof DerefSymbol) {
            ret = new UnaryExpression(UnaryOperator.DEREFERENCE,
                    ((DerefSymbol)ref_symbol).toExpression());
        } else {
            ret = new UnaryExpression(UnaryOperator.DEREFERENCE,
                    new Identifier(ref_symbol));
        }
        return ret;
    }

    /**
    * Checks if this deref symbol equals to the given object.
    * @param o the object to be compared with.
    * @return true if they are equal, false otherwise.
    */
    public boolean equals(Object o) {
        return (o != null &&
                o instanceof DerefSymbol &&
                ref_symbol.equals(((DerefSymbol)o).ref_symbol));
    }

    /**
    * Returns the hash code of the current deref symbol.
    */
    public int hashCode() {
        return (ref_symbol.hashCode() ^ toString().hashCode());
    }

}
