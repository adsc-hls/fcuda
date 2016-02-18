package cetus.hir;

/**
* Thrown when a symbol is about to be entered into a symbol table and there is
* already a symbol of the same name in that same table.
*/
public class DuplicateSymbolException extends RuntimeException {

    private static final long serialVersionUID = 3477L;

    /**
    * Creates the exception.
    *
    * @param message The message should indicate the name of the offending
    *       symbol.
    */
    public DuplicateSymbolException(String message) {
        super(message);
    }

}
