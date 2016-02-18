package cetus.hir;

import java.util.List;
import java.util.Set;

public interface SymbolTable extends Traversable {

    /**
    * Add a declaration to the end of the set of declarations
    * and place the declared symbols in the symbol table.
    *
    * @param decl The declaration to add.
    */
    void addDeclaration(Declaration decl);

    /**
    * Add a declaration before the reference declaration.
    *
    * @param ref The reference point.
    * @param decl The declaration to add.
    */
    void addDeclarationBefore(Declaration ref, Declaration decl);

    /**
    * Add a declaration after the reference declaration.
    *
    * @param ref The reference point.
    * @param decl The declaration to add.
    */
    void addDeclarationAfter(Declaration ref, Declaration decl);

    /**
    * Retrieves the declaration for a symbol, possibly searching
    * through parent symbol tables.
    *
    * @param name The name of the symbol.
    */
    Declaration findSymbol(IDExpression name);

    /**
    * Returns a list of symbol tables that are parents
    * of this table in the distributed symbol table graph.
    * A list is necessary because representing multiple
    * inheritence in C++ requires it.  Note that the parent
    * tables are not necessarily tables of the immediate
    * parent.  The list is all symbol tables enclosing this
    * object which appear at the same level such that no
    * other symbol tables appear between this object and
    * those tables.
    *
    * @return a List of SymbolTables.
    */
    List<SymbolTable> getParentTables();

    /**
    * Checks if the symbol table contains the specified declaration.
    *
    * @return true if the symbol table contains <b>decl</b>.
    */
    boolean containsDeclaration(Declaration decl);

    /**
    * Checks if the symbol table contains the specified symbol.
    *
    * @return true if the symbol table contains <b>symbol</b>.
    */
    boolean containsSymbol(Symbol symbol);

    /**
    * Returns the set of declared symbols in the symbol table object.
    *
    * @return the set of declared symbols.
    */
    Set<Symbol> getSymbols();

    /**
    * Returns the set of declarations declared within the symbol table object.
    *
    * @return the set of declarations within the symbol table object.
    */
    Set<Declaration> getDeclarations();

    /** TODO
    * Removes the specified child declaration from the symbol table object.
    */
    //void removeDeclaration(Declaration decl);

    /** TODO
    * Removes the specified symbol object from the symbol table. This modifier
    * consistently removes the specified symbol object from the IR and from the
    * internal look-up table.
    */
    //void removeSymbol(Symbol symbol);

}
