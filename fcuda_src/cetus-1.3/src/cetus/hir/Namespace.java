package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Set;

/**
* Represents a block of declarations that share a common namespace.
* This class is not used in C programs.
*/
/** This class is not supported */
public class Namespace extends Declaration implements SymbolTable {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Namespace.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private IDExpression name;

    private Map<IDExpression, Declaration> symbol_table;

    /**
    * Creates an anonymous namespace.
    */
    public Namespace() {
        object_print_method = class_print_method;
        name = null;
    }

    /**
    * Creates a named namespace.
    *
    * @param name The name of the namespace.
    */
    public Namespace(IDExpression name) {
        object_print_method = class_print_method;
        this.name = name;
    }

    public void addDeclaration(Declaration decl) {
        children.add(decl);
        // TODO - symbol table & decl stmt
    }

    public void addDeclarationBefore(Declaration ref, Declaration decl) {
    }

    public void addDeclarationAfter(Declaration ref, Declaration decl) {
    }

    /**
    * Prints a namespace to a stream.
    *
    * @param n The namespace to print.
    * @param o The writer on which to print the namespace.
    */
    public static void defaultPrint(Namespace n, PrintWriter o) {
        o.print("namespace ");
        if (n.name != null) {
            n.name.print(o);
        }
        o.println("\n{");
        PrintTools.printlnList(n.children, o);
        o.print("}");
    }

    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    public List<IDExpression> getDeclaredIDs() {
        List<IDExpression> ret = new ArrayList<IDExpression>(1);
        ret.add(name);
        return ret;
    }

    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    *
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    public Set<Symbol> getSymbols() {
        return SymbolTools.getSymbols(this);
    }

    public Set<Declaration> getDeclarations() {
        return new LinkedHashSet<Declaration>(symbol_table.values());
    }

    /* SymbolTable interface */
    public boolean containsSymbol(Symbol symbol) {
        for (IDExpression id : symbol_table.keySet()) {
            if (id instanceof Identifier &&
                symbol.equals(((Identifier) id).getSymbol())) {
                return true;
            }
        }
        return false;
    }

    /* SymbolTable interface */
    public boolean containsDeclaration(Declaration decl) {
        return symbol_table.containsValue(decl);
    }

}
