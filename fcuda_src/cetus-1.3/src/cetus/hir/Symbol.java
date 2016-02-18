package cetus.hir;

import java.util.List;

/**
* An IR object that implements Symbol interface is identified as a unique
* symbol in the program. Every {@link Identifier} object has a link to
* its corresponding Symbol object and can access the attributes of the symbol
* object.
*/
public interface Symbol {

    /**
    * Returns a list of type specifiers.
    * 
    * @return the list of type specifiers.
    */
    List getTypeSpecifiers();

    /**
    * Returns a list of array specifiers.
    *
    * @return the list of array specifiers.
    */
    List getArraySpecifiers();

    /**
    * Returns the name of the symbol.
    *
    * @return the name of the symbol.
    */
    String getSymbolName();

    /**
    * Modify the name of the symbol. This method effectively renames all the
    * identifiers constructed from the symbol object. Pass writers should be
    * cautious when renaming global names since the effective scope of this
    * operation is within a translation unit; {@code extern} variables and
    * procedure names are not automatically handled.
    *
    * @param name the new name of the symbol.
    */
    void setName(String name);

    /**
    * Returns the parent declaration containing the symbol object. The return
    * value is searched from the internal look-up table. Depending on the type
    * of the symbol, this method may return the symbol object itself, or null
    * if the symbol object does not exist in the IR.
    *
    * @return the parent declaration.
    */
    Declaration getDeclaration();

}
