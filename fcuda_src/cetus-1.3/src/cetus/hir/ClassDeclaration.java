package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;

/**
* Represents a class, struct, or union.  These are actually specifiers in C and
* C++, since variables can be declared immediately following the class
* declaration, but that's mostly syntactic sugar.  We assume it can be split
* into a class declaration followed by a variable declaration.  We have a single
* class representing all three (class, struct, union) because they are
* fundamentally the same except for default access levels and storage layout,
* which do not make a significant difference to source-to-source compilers.
* Anonymous structs should be given a unique name (most compilers do this
* internally anyway).
*/
public class ClassDeclaration extends Declaration implements SymbolTable {

    /** The default class print method */
    private static Method class_print_method;

    /** The default method for printing the class declaration in C++ */
    protected static final Method print_as_cpp;

    /** The default method for printing the class declaration in JAVA */
    protected static final Method print_as_java;

    /** Assigns the default print methods. */
    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ClassDeclaration.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
            print_as_cpp = params[0].getMethod("printCpp", params);
            print_as_java = params[0].getMethod("printJava", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError(e.getMessage());
        }
    }

    /**
    * Class for representing the four different types of class declarations,
    * <b>class</b>, <b>struct</b>, <b>union</b>, and <b>interface</b>.
    */
    public static class Key {

        /** Four predefined type keywords */
        private static final String[] name = {
                "class", "struct", "union", "interface"};

        /** The keyword for the Key object */
        private int value;

        /** Constructs a type key from the predefined number */
        private Key(int value) {
            this.value = value;
        }

        /** Prints the type keyword */
        private void print(PrintWriter o) {
            o.print(name[value]);
        }

        /** Returns the string representation of the type keyword */
        @Override
        public String toString() {
            return name[value];
        }

    }

    /** Keyword for <b>class</b> type declaration. */
    public static final Key CLASS = new Key(0);

    /** Keyword for <b>struct</b> type declaration. */
    public static final Key STRUCT = new Key(1);

    /** Keyword for <b>union</b> type declaration. */
    public static final Key UNION = new Key(2);

    /** Keyword for <b>interface</b> type declaration. */
    public static final Key INTERFACE = new Key(3);

    /**
    * For Java only: leading specifiers on the class declaration itself, such
    * as public or abstract.
    */
    private List class_specs;

    /** The keyword for the class declaration */
    private Key type;

    /** The name ID for the class declaration */
    private IDExpression name;

    /**
    * For C++ only: Paired one-to-one with extends_list to specify the type of
    * inheritence.
    */
    private List extends_access_list;

    /** List of classes that this class extends. */
    private List extends_list;

    /** List of interfaces that this class implements. */
    private List implements_list;

    /** Internal look-up table */
    private Map<IDExpression, Declaration> symbol_table =
            new LinkedHashMap<IDExpression, Declaration>(4);

    /**
    * Constructs a class declaration from the given list of specifiers, type
    * keyword, and name ID. This method is intended for JAVA classes.
    */
    public ClassDeclaration(List class_specs, Key type, IDExpression name) {
        object_print_method = class_print_method;
        this.class_specs = class_specs;
        this.type = type;
        this.name = new NameID(name.toString());
        extends_access_list = new ArrayList(1);
        extends_list = new ArrayList(1);
        implements_list = new ArrayList(1);
    }

    /**
    * Constructs an empty class declaration with the given type and name ID.
    *
    * @param type Must be one of CLASS, STRUCT, UNION, or INTERFACE.
    * @param name The name ID for the class.
    */
    public ClassDeclaration(Key type, IDExpression name) {
        object_print_method = class_print_method;
        class_specs = new ArrayList(1);
        this.type = type;
        this.name = new NameID(name.toString());
        extends_access_list = new ArrayList(1);
        extends_list = new ArrayList(1);
        implements_list = new ArrayList(1);
    }

    /**
    * Constructs a class declaration with the given type, name ID, and the flag
    * for <b>forward declaration</b>.
    *
    * @param type Must be one of CLASS, STRUCT, UNION, or INTERFACE.
    * @param name The name for the class.
    * @param no_body True if this a forward declaration.
    */
    public ClassDeclaration(Key type, IDExpression name, boolean no_body) {
        object_print_method = class_print_method;
        class_specs = new ArrayList(1);
        this.type = type;
        this.name = new NameID(name.toString());
        extends_access_list = new ArrayList(1);
        extends_list = new ArrayList(1);
        implements_list = new ArrayList(1);
        if (no_body) {
            children = null;
        }
    }

    /** Adds the name of the base class - not used in C */
    @SuppressWarnings("unchecked")
    public void addBaseClass(IDExpression name) {
        extends_access_list.add(null);
        extends_list.add(name);
    }

    /** Adds the name of the base class with the specifier - not used in C */
    @SuppressWarnings("unchecked")
    public void addBaseClass(Specifier access, IDExpression name) {
        extends_access_list.add(access);
        extends_list.add(name);
    }

    /** Adds the name of the interface - not used in C */
    @SuppressWarnings("unchecked")
    public void addBaseInterface(IDExpression name) {
        implements_list.add(name);
    }

    /* SymbolTable interface */
    public void addDeclaration(Declaration decl) {
        if (children == null) {
            throw new IllegalStateException();
        }
        if (decl instanceof VariableDeclaration ||
            decl instanceof Enumeration) {
            DeclarationStatement stmt = new DeclarationStatement(decl);
            children.add(stmt);
            stmt.setParent(this);
        } else {
            children.add(decl);
            decl.setParent(this);
        }
        SymbolTools.addSymbols(this, decl);
    }

    /* SymbolTable interface */
    public void addDeclarationBefore(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            if (ref.getParent() instanceof DeclarationStatement) {
                index = Tools.identityIndexOf(children, ref.getParent());
            }
            if (index == -1) {
                throw new IllegalArgumentException();
            }
        }
        if (decl instanceof VariableDeclaration ||
            decl instanceof Enumeration) {
            DeclarationStatement stmt = new DeclarationStatement(decl);
            children.add(index, stmt);
            stmt.setParent(this);
        } else {
            children.add(index, decl);
            decl.setParent(this);
        }
        SymbolTools.addSymbols(this, decl);
    }

    /* SymbolTable interface */
    public void addDeclarationAfter(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            if (ref.getParent() instanceof DeclarationStatement) {
                index = Tools.identityIndexOf(children, ref.getParent());
            }
            if (index == -1) {
                throw new IllegalArgumentException();
            }
        }
        if (decl instanceof VariableDeclaration ||
            decl instanceof Enumeration) {
            DeclarationStatement stmt = new DeclarationStatement(decl);
            // if ref is last child
            if (index == children.size() - 1) {
                children.add(stmt);
            } else {
                children.add(index + 1, stmt);
            }
            stmt.setParent(this);
        } else {
            // if ref is last child
            if (index == children.size() - 1) {
                children.add(decl);
            } else {
                children.add(index + 1, decl);
            }
            decl.setParent(this);
        }
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * Prints a class to a stream.
    *
    * @param d The class to print.
    * @param o The writer on which to print the class.
    */
    public static void defaultPrint(ClassDeclaration d, PrintWriter o) {
        printCpp(d, o);
    }

    /* SymbolTable interface */
    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    /**
    * Returns the IDs declared by this class declaration; the returned list
    * contains one ID which gives the type and the name. e.g., "struct foo".
    *
    * @return the list of the declared IDs.
    */
    public List<IDExpression> getDeclaredIDs() {
        List<IDExpression> list = new ArrayList<IDExpression>(1);
        list.add(new NameID(type + " " + name));
        return list;
    }

    /** Returns the type keyword for the class declaration */
    public Key getKey() {
        return type;
    }

    /** Returns the name ID of the class declaration */
    public IDExpression getName() {
        return name;
    }

    /* SymbolTable interface */
    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    /**
    * Returns the direct handle to the internal look-up table for symbols.
    * This method is protected for consistent management of symbols.
    *
    * @return the internal look-up table for symbols.
    */
    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
    }

    /**
    * Prints a C++ class to a stream.
    *
    * @param d The class to print.
    * @param o The writer on which to print the class.
    */
    public static void printCpp(ClassDeclaration d, PrintWriter o) {
        d.type.print(o);
        o.print(" ");
        d.name.print(o);
        if (!d.extends_access_list.isEmpty() || !d.extends_list.isEmpty()) {
            o.print(" : ");
            Iterator iter = d.extends_access_list.iterator();
            Iterator iter2 = d.extends_list.iterator();
            while (iter.hasNext()) {
                Specifier as = (Specifier)iter.next();
                IDExpression id = (IDExpression)iter.next();
                if (as != null) {
                    as.print(o);
                }
                id.print(o);
            }
        }
        if (d.children != null) {
            o.println("");
            o.println("{");
            PrintTools.printlnList(d.children, o);
            o.println("};");
        } else {
            o.println(";");
        }
    }

    /**
    * Prints a Java class to a stream.
    *
    * @param d The class to print.
    * @param o The writer on which to print the class.
    */
    public static void printJava(ClassDeclaration d, PrintWriter o) {
        PrintTools.printListWithSeparator(d.class_specs, o, " ");
        o.print(" ");
        d.type.print(o);
        o.print(" ");
        d.name.print(o);
        if (!d.extends_list.isEmpty()) {
            o.print(" extends ");
            o.print(d.extends_list.get(0).toString());
        }
        if (!d.implements_list.isEmpty()) {
            o.print(" implements ");
            PrintTools.printListWithComma(d.implements_list, o);
        }
        if (d.children != null) {
            o.println("\n{");
            PrintTools.printlnList(d.children, o);
            o.println("}");
        } else {
            o.println(";");
        }
    }

    /* Traversable interface */
    public void removeChild(Traversable child) {
        int index = Tools.identityIndexOf(children, child);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        children.remove(index);
        child.setParent(null);
        if (child instanceof Declaration) {
            SymbolTools.removeSymbols(this, (Declaration) child);
        } else if (child instanceof DeclarationStatement) {
            SymbolTools.removeSymbols(this,
                    ((DeclarationStatement)child).getDeclaration());
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
    * Returns a clone of this class declaration. The features for C++ are not
    * supported (access specifications, ...).
    *
    * @return the cloned class declaration.
    */
    @Override
    @SuppressWarnings("unchecked")
    public ClassDeclaration clone() {
        ClassDeclaration o = (ClassDeclaration)super.clone();
        o.type = type;
        o.name = name.clone();
        o.class_specs = new ArrayList(class_specs.size());
        o.extends_access_list = new ArrayList(extends_access_list.size());
        o.extends_list = new ArrayList(extends_list.size());
        o.implements_list = new ArrayList(extends_list.size());
        o.symbol_table = new LinkedHashMap<IDExpression, Declaration>(4);
        for (Object obj : class_specs) {
            // obj should be cloned, but we don't know what it is. doing the
            // same for others as well.
            o.class_specs.add(obj);
        }
        for (Object obj : extends_list) {
            o.extends_list.add(obj);
        }
        for (Object obj : extends_access_list) {
            o.extends_access_list.add(obj);
        }
        for (Object obj : implements_list) {
            o.implements_list.add(obj);
        }
        // Builds the look-up table by traversing the children.
        DepthFirstIterator iter = new DepthFirstIterator(o);
        iter.next();            // skip the class declaration itself.
        iter.pruneOn(Declaration.class);
        while (iter.hasNext()) {
            Object child = iter.next();
            if (child instanceof Declaration) {
                SymbolTools.addSymbols(o, (Declaration)child);
            }
        }
        return o;
    }

    // TODO: This violates Declaration's equals() method - within a single
    // translation unit, it is more natural to provide object comparison rather
    // than contents comparison. Maybe introducing "match()" method is better
    // idea.
    public boolean matches(Object otherObj) {
        if (this == otherObj) {
            return true;
        }
        if (!(otherObj instanceof ClassDeclaration)) {
            return false;
        }
        ClassDeclaration other = (ClassDeclaration)otherObj;
        if (!this.name.equals(other.name)) {
            return false;
        }
        if (!this.type.equals(other.type)) {
            return false;
        }
        // extends_list, extends_access_list and implements_list don't seem to
        // be used so we are not checking them for equality, for now
        if (this.children.size() != other.children.size()) {
            return false;
        }
        for (int i = 0; i < this.children.size(); i++) {
            // to make the comparison easy, lets just compare the results of
            // toString()... not a good approach, may change later on
            if (!this.children.get(i).toString().equals(
                        other.children.get(i).toString())) {
                return false;
            }
        }
        return true;
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
