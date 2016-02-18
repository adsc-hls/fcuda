package cetus.hir;

import cetus.exec.Driver;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

/**
* Represents a single source file of the program.
* A TranslationUnit is a collection of Declarations. The compilation of a
* Program may involve several TranslationUnits.
*/
public final class TranslationUnit implements SymbolTable, Traversable {

    private static Method class_print_method;

    private Method object_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = TranslationUnit.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The name of the original source file. */
    private String input_filename;

    /** The name of the modified source file. */
    private String output_filename;

    /** The parent Program object. */
    private Traversable parent;

    /** A list of Declarations and DeclarationStatements. */
    private List<Traversable> children;

    /** Maps IDExpressions to Declarations. */
    private Map<IDExpression, Declaration> symbol_table;

    /**
    * Create an empty translation unit associated with a file.
    *
    * @param input_filename The file name for this TranslationUnit
    */
    public TranslationUnit(String input_filename) {
        object_print_method = class_print_method;
        this.input_filename = input_filename;
        // output_filename just keeps its name not path name.
        output_filename = (new File(input_filename)).getName();
        parent = null;
        children = new ArrayList<Traversable>();
        symbol_table = new LinkedHashMap<IDExpression, Declaration>();
    }

    /* SymbolTable interface */
    public void addDeclaration(Declaration decl) {
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (decl instanceof VariableDeclaration || decl instanceof Enumeration){
            decl.setSemiColon(true);
        }
        children.add(decl);
        decl.setParent(this);
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * Adds a new declaration <b>decl</b> at the head of the child list.
    *
    * @param decl the new declaration to be added.
    * @throws NotAnOrphanException if <b>decl</b> has a parent object.
    */
    public void addDeclarationFirst(Declaration decl) {
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (decl instanceof VariableDeclaration || decl instanceof Enumeration){
            decl.setSemiColon(true);
        }
        children.add(0, decl);
        decl.setParent(this);
        SymbolTools.addSymbols(this, decl);
    }

    /* SymbolTable interface */
    public void addDeclarationBefore(Declaration ref, Declaration decl) {
        int index = children.indexOf(ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl instanceof VariableDeclaration || decl instanceof Enumeration){
            decl.setSemiColon(true);
        }
        children.add(index, decl);
        decl.setParent(this);
        SymbolTools.addSymbols(this, decl);
    }

    /* SymbolTable interface */
    public void addDeclarationAfter(Declaration ref, Declaration decl) {
        int index = children.indexOf(ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl instanceof VariableDeclaration || decl instanceof Enumeration){
            decl.setSemiColon(true);
        }
        children.add(index + 1, decl);
        decl.setParent(this);
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * Returns the first declaration that comes right after
    * header files.
    * 
    * @return First declaration that comes after header files. 
    */
    public Declaration getFirstDeclaration() {
        Declaration firstdecl = null;
        boolean foundHeaderEnd = false;
        int children_size = children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = children.get(i);
            if (foundHeaderEnd) {
                if (child instanceof Declaration) {
                    firstdecl = (Declaration)child;
                } else {
                    System.err.println(
                            "cetus: Unknown class type in TranslationUnit");
                    Tools.exit(1);
                }
                foundHeaderEnd = false;
            }
            if (child instanceof AnnotationDeclaration) {
                String note = child.toString();
                if (note.startsWith("#pragma endinclude")) {
                    foundHeaderEnd = true;
                }
            }
        }
        if (firstdecl == null) {
            Object firstchild = children.get(0);
            if (firstchild instanceof Declaration) {
                firstdecl = (Declaration)firstchild;
            } else {
                System.err.println(
                        "cetus: Unknown class type in TranslationUnit");
                Tools.exit(1);
            }
        }
        return firstdecl;
    }

    /** Returns a string representation of this translation unit. */
    @Override public String toString() {
        StringWriter sw = new StringWriter(80000);
        print(new PrintWriter(sw));
        return sw.toString();
    }

    /* SymbolTable interface */
    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    /* Traversable interface */
    public List<Traversable> getChildren() {
        return children;
    }

    /**
    * Returns the original filename for the translation unit.
    *
    * @return the original filename for the translation unit.
    */
    public String getInputFilename() {
        return input_filename;
    }

    /**
    * Returns the new filename for the translation unit to be written to.
    *
    * @return the new filename.
    */
    public String getOutputFilename() {
        return output_filename;
    }

    /* Traversable interface */
    public Traversable getParent() {
        return parent;
    }

    /* SymbolTable interface */
    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    /**
    * Returns the internal look-up table for symbols. This method is protected
    * for consistent symbol table management.
    *
    * @return the internal look-up table.
    */
    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
    }

    /**
    * Prints a translation unit to a stream without skipping the included
    * headers.
    * @param t The translation unit to print.
    * @param o The writer on which to print the translation unit.
    */
    public static void defaultPrint2(TranslationUnit t, PrintWriter o) {
        boolean now_skipping_header = false;
        String expand_user = Driver.getOptionValue("expand-user-header");
        String expand_all = Driver.getOptionValue("expand-all-header");
        int children_size = t.children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = t.children.get(i);
            if (child instanceof AnnotationDeclaration) {
                String note = child.toString();
                if (note.startsWith("#pragma startinclude") &&
                    (expand_user == null || !note.contains("\"")) &&
                    (expand_all == null || !note.contains("\"") &&
                    !note.contains("<"))) {  // every header is skipped
                    o.println(note.replace("#pragma startinclude ", ""));
                    now_skipping_header = true;
                } else if (note.startsWith("#pragma endinclude")) {
                    // reset the skip flag
                    now_skipping_header = false;
                } else if (!now_skipping_header) {
                    // normal printing of annotation declaration
                    child.print(o);
                    o.println("");
                }
            } else if (!now_skipping_header) {
                // normal printing of the child
                child.print(o);
                o.println("");
            }
        }
    }

    public static void defaultPrint(TranslationUnit tu, PrintWriter o) {
        final int USER_HEADER = 1;
        final int SYS_HEADER = 2;
        int expand = 0;
        if (Driver.getOptionValue("expand-user-header") != null) {
            expand = USER_HEADER;
        }
        if (Driver.getOptionValue("expand-all-header") != null) {
            expand = SYS_HEADER;
        }
        List<Integer> state = new LinkedList<Integer>();
        int children_size = tu.children.size();
        for (int i = 0; i < children_size; i++) {
            Traversable child = tu.children.get(i);
            boolean skip = false;
            PragmaAnnotation p = ((Annotatable)child).getAnnotation(
                    PragmaAnnotation.class, "pragma");
            if (p != null) {
                String p_name = p.getName();
                if (p_name.startsWith("startinclude")) {
                    if (p_name.contains("\"")) {
                        state.add(0, USER_HEADER);
                    } else {
                        state.add(0, SYS_HEADER);
                    }
                    if (state.get(0) > expand) {
                        o.println(p_name.replace("startinclude ", ""));
                    } else {
                        child.print(o);
                        o.println("");
                    }
                } else if (p_name.startsWith("endinclude")) {
                    if (state.get(0) <= expand) {
                        child.print(o);
                        o.println("");
                    }
                    state.remove(0);
                }
                continue;
            }
            if (!state.isEmpty() && state.get(0) > expand) {
                continue;
            }
            child.print(o);
            o.println("");
        }
    }

    /**
    * Prints this translation unit to the output file with which it is
    * associated.
    *
    * @param outDir the target directory.
    * @throws FileNotFoundException if an output file could not be opened.
    */
    public void print(File outDir) throws FileNotFoundException {
        File to = new File(outDir, output_filename);
        try {
            // default buffer size 8192 (characters).
            PrintWriter o = new PrintWriter(
                    new BufferedWriter(new FileWriter(to)));
             print(o);
             o.close();
        } catch(IOException e) {
            throw new FileNotFoundException(e.getMessage());
        }
    }

    /**
    * Prints this translation unit to the given print writer.
    *
    * @param o the output print writer.
    */
    public void print(PrintWriter o) {
        if (object_print_method == null) {
            return;
        }
        try {
            object_print_method.invoke(null, new Object[] {this, o});
        } catch(IllegalAccessException e) {
            throw new InternalError(e.getMessage());
        } catch(InvocationTargetException e) {
            throw new InternalError(e.getMessage());
        }
    }

    /* Traversable interface */
    public void removeChild(Traversable child) {
        int index = Tools.identityIndexOf(children, child);
        if (index == -1) {
            throw new NotAChildException();
        }
        child.setParent(null);
        children.remove(index);
        // removing a declaration removes the declared symbols
        if (child instanceof Declaration) {
            SymbolTools.removeSymbols(this, (Declaration)child);
        }
    }

    /* Traversable interface */
    public void setChild(int index, Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException();
        }
        // detach the old child at position index and remove any declared
        // symbols
        Traversable old = children.get(index);
        old.setParent(null);
        if (old instanceof Declaration) {
            SymbolTools.removeSymbols(this, (Declaration)old);
        }
        if (t instanceof VariableDeclaration || t instanceof Enumeration) {
            ((Declaration)t).setSemiColon(true);
        }
        if (t instanceof Declaration) {
            children.set(index, t);
            t.setParent(this);
        } else {
            throw new IllegalArgumentException();
        }
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

    /**
    * Sets the output filename for the translation unit.
    *
    * @param output_filename The output filename for the translation unit.
    */
    public void setOutputFilename(String output_filename) {
        this.output_filename = output_filename;
    }

    /**
    * Sets the parent program for this translation unit.
    * @param t The parent program.
    */
    public void setParent(Traversable t) {
        if (!t.getChildren().contains(this)) {
            throw new NotAChildException();
        }
        if (t instanceof Program) {
            parent = t;
        } else {
            throw new IllegalArgumentException();
        }
    }

    /**
    * Overrides the print method for this object only.
    *
    * @param m The new print method.
    */
    public void setPrintMethod(Method m) {
        object_print_method = m;
    }

    /* SymbolTable interface */
    public Set<Symbol> getSymbols() {
        return SymbolTools.getSymbols(this);
    }

    /* SymbolTable interface */
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
