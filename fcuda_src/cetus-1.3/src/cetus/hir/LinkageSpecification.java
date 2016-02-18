package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;

/**
* Represents the <var>extern "C"</var> specification
* in C++ programs. This class is not used in C programs.
*/
/** This class is not supported */
public class LinkageSpecification extends Declaration implements SymbolTable {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = LinkageSpecification.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private String calling_convention;

    private Map<IDExpression, Declaration> symbol_table;

    /**
    * Represents <var>extern "s" decl</var>.
    *
    * @param s The name of the other language.
    * @param decl The declaration whose linkage is being specified.
    */
    public LinkageSpecification(String s, Declaration decl) {
        super(1);
        object_print_method = class_print_method;
        calling_convention = s;
        children.add(decl);
    }

    /**
    * Represents <var>extern "s" { decl_list }</var>.
    *
    * @param s The name of the other language.
    * @param decl_list The declaration whose linkage is being specified.
    */
    public LinkageSpecification(String s, List<Traversable> decl_list) {
        super(decl_list.size());
        object_print_method = class_print_method;
        calling_convention = s;
        children.addAll(decl_list);
    }

    public void addDeclaration(Declaration decl) {
        children.add(decl);
        // TODO - decl stmt & symbol table entry
    }

    public void addDeclarationBefore(Declaration ref, Declaration decl) {
    }

    public void addDeclarationAfter(Declaration ref, Declaration decl) {
    }

    /**
    * Prints a linkage specification block to a stream.
    *
    * @param s The block to print.
    * @param o The writer on which to print the block.
    */
    public static void defaultPrint(LinkageSpecification s, PrintWriter o) {
        o.print("extern \"");
        o.print(s.calling_convention);
        o.println("\"\n{");
        PrintTools.printlnList(s.children, o);
        o.print("}");
    }

    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    @SuppressWarnings("unchecked")
    public List<IDExpression> getDeclaredIDs() {
        return (List<IDExpression>)empty_list;
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
                symbol.equals(((Identifier)id).getSymbol())) {
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
