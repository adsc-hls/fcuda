package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;

/** Represents a declarator for a variable in a VariableDeclaration. */
public class VariableDeclarator extends Declarator implements Symbol {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = VariableDeclarator.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** Common initialization process for constructors. */
    private void initialize(IDExpression direct_decl) {
        object_print_method = class_print_method;
        if (direct_decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        // Forces use of NameID instead of Identifier.
        if (!(direct_decl instanceof NameID)) {
            direct_decl = new NameID(direct_decl.toString());
        }
        children.add(direct_decl);
        direct_decl.setParent(this);
    }

    /**
    * Constructs a new VariableDeclarator with the given ID.
    * It is highly recommended to use a {@link NameID} object for
    * <b>direct_decl</b> since the constructor internally assigns a new
    * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
    *
    * @param direct_decl the given name ID for the new variable declarator.
    */
    public VariableDeclarator(IDExpression direct_decl) {
        initialize(direct_decl);
        leading_specs = new ArrayList<Specifier>(1);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new variable declarator with the given ID and the trailing
    * specifiers.
    * It is highly recommended to use a {@link NameID} object for
    * <b>direct_decl</b> since the constructor internally assigns a new
    * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
    *
    * @param direct_decl the given name ID.
    * @param trailing_specs the list of trailing specifiers.
    */
    @SuppressWarnings("unchecked")
    public VariableDeclarator(IDExpression direct_decl, List trailing_specs) {
        initialize(direct_decl);
        this.leading_specs = new ArrayList<Specifier>(1);
        this.trailing_specs = new ArrayList<Specifier>(trailing_specs);
    }

    /**
    * Constructs a new variable declarator with the given name ID and the
    * trailing specifier.
    * It is highly recommended to use a {@link NameID} object for
    * <b>direct_decl</b> since the constructor internally assigns a new
    * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
    *
    * @param direct_decl the given name ID.
    * @param spec the given trailing specifier.
    */
    public VariableDeclarator(IDExpression direct_decl, Specifier spec) {
        initialize(direct_decl);
        this.leading_specs = new ArrayList<Specifier>(1);
        this.trailing_specs = new ArrayList<Specifier>(1);
        this.trailing_specs.add(spec);
    }

    /**
    * Constructs a new variable declarator with the given leading specifiers and
    * the name ID.
    * It is highly recommended to use a {@link NameID} object for
    * <b>direct_decl</b> since the constructor internally assigns a new
    * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
    *
    * @param leading_specs the list of leading specifiers.
    * @param direct_decl the given name ID.
    */
    @SuppressWarnings("unchecked")
    public VariableDeclarator(List leading_specs, IDExpression direct_decl) {
        initialize(direct_decl);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new variable declarator with the given leading specifiers,
    * the name ID, and the trailing specifiers.
    * It is highly recommended to use a {@link NameID} object for
    * <b>direct_decl</b> since the constructor internally assigns a new
    * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
    *
    * @param leading_specs the list of leading specifiers.
    * @param direct_decl the given name ID.
    * @param trailing_specs the list of trailing specifiers.
    */
    @SuppressWarnings("unchecked")
    public VariableDeclarator(List leading_specs, IDExpression direct_decl,
                              List trailing_specs) {
        initialize(direct_decl);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(trailing_specs);
    }

    /**
    * Constructs a new variable declarator with the given leading specifier and
    * the name ID.
    * It is highly recommended to use a {@link NameID} object for
    * <b>direct_decl</b> since the constructor internally assigns a new
    * <b>NameID</b> if <b>direct_decl</b> is not an instanceof <b>NameID</b>.
    *
    * @param spec the given leading specifier.
    * @param direct_decl the given name ID.
    */
    public VariableDeclarator(Specifier spec, IDExpression direct_decl) {
        initialize(direct_decl);
        this.leading_specs = new ArrayList<Specifier>(1);
        this.leading_specs.add(spec);
        this.trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Returns a clone of this variable declarator.
    */
    @Override
    public VariableDeclarator clone() {
        VariableDeclarator d = (VariableDeclarator)super.clone();
        if (children.size() > 0) {
            IDExpression id = getDirectDeclarator().clone();
            d.children.add(id);
            id.setParent(d);
            if (getInitializer() != null) {
                Initializer init = getInitializer().clone();
                d.setInitializer(init);
            }
        }
        return d;
    }

    /**
    * Prints a variable declarator to a stream.
    *
    * @param d The declarator to print.
    * @param o The writer on which to print the declarator.
    */
    public static void defaultPrint(VariableDeclarator d, PrintWriter o) {
        if (!d.leading_specs.isEmpty()) {
            PrintTools.printListWithSpace(d.leading_specs, o);
            //o.print(" ");
        }
        d.getDirectDeclarator().print(o);
        if (!d.trailing_specs.isEmpty()) {
            PrintTools.printListWithSpace(d.trailing_specs, o);
        }
        if (d.getInitializer() != null) {
            d.getInitializer().print(o);
        }
    }

    /**
    * Returns the name ID of this variable declarator.
    */
    protected IDExpression getDirectDeclarator() {
        return (IDExpression)children.get(0);
    }

    /**
    * Returns the name ID of this variable declarator.
    */
    public IDExpression getID() {
        return getDirectDeclarator();
    }

    /**
    * Returns the list of leading specifiers of this variable declarator.
    */
    @SuppressWarnings("unchecked")
    public List getSpecifiers() {
        return leading_specs;
    }

    /**
    * Returns the list of trailing specifiers of this variable declarator.
    */
    public List getTrailingSpecifiers() {
        return trailing_specs;
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
    * Returns the initializer of this variable declarator.
    *
    * @return the initializer if it exists, null otherwise.
    */
    public Initializer getInitializer() {
        if (children.size() > 1) {
            return (Initializer)children.get(1);
        } else {
            return null;
        }
    }

    /**
    * Assigns a new initializer <b>init</b> for the variable declarator.
    * The existing initializer is discarded if exists.
    */
    public void setInitializer(Initializer init) {
        if (getInitializer() != null) {
            getInitializer().setParent(null);
            if (init != null) {
                children.set(1, init);
                init.setParent(this);
            } else {
                children.remove(1);
            }
        } else {
            if (init != null) {
                children.add(init);
                init.setParent(this);
            }
        }
    }

    /* Symbol interface */
    public String getSymbolName() {
        return getID().toString();
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getTypeSpecifiers() {
        Traversable t = this;
        while (!(t instanceof Declaration)) {
            t = t.getParent();
        }
        List ret = new ArrayList();
        if (t instanceof VariableDeclaration) {
            ret.addAll(((VariableDeclaration)t).getSpecifiers());
        } else if (t instanceof Enumeration) {
            ret.add(((Enumeration)t).getSpecifier());
        } else {
            return null;
        }
        ret.addAll(leading_specs);
        return ret;
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getArraySpecifiers() {
        return trailing_specs;
    }

    /** Sets the direct declarator of the variable declarator */
    @Override
    protected void setDirectDeclarator(IDExpression id) {
        if (id.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.get(0).setParent(null);
        children.set(0, id);
        id.setParent(this);
    }

    /* Symbol interface */
    public void setName(String name) {
        SymbolTable symtab = IRTools.getAncestorOfType(this, SymbolTable.class);
        SymbolTools.setSymbolName(this, name, symtab);
    }

    /* Symbol interface */
    public Declaration getDeclaration() {
        return IRTools.getAncestorOfType(this, Declaration.class);
    }

    public List<Declaration> getParameters() {
        throw new UnsupportedOperationException();
    }

    public void addParameter(Declaration decl) {
        throw new UnsupportedOperationException();
    }

    public void addParameterAfter(Declaration ref, Declaration decl) {
        throw new UnsupportedOperationException();
    }

    public void addParameterBefore(Declaration ref, Declaration decl) {
        throw new UnsupportedOperationException();
    }

  /* *AP*
    Adding method for removing parameters from declarator list
  */
  public void removeParameter(Declaration decl)
  {
    throw new UnsupportedOperationException();
  }


    /**
    * Warning: this is an experimental feature.
    * Checks if the given Object <b>obj</b> refers to the same Symbol as this
    * Symbol.
    * @param obj
    * @return true if <b>obj</b> refers to the same memory location as this
    * Symbol.
    */
    private boolean isSameSymbol(Object obj) {
        // obj should be of type Symbol
        if (!(obj instanceof Symbol)) {
            return false;
        }
        Symbol other = (Symbol)obj;
        // Check text value of both Identifiers
        if (!this.getSymbolName().equals(other.getSymbolName())) {
            return false;
        }
        // If symbols are equal, two Identifiers refer to the same storage
        if (this == other) {
            return true;
        } else {
            // If both symbols
            // 1. Have the same text value and
            // 2. Have external linkage
            return hasExternalLinkage(this) && hasExternalLinkage(other);
        }
    }

    /**
    * Check if Symbol <b>sym</b> has external linkage.
    * @param sym
    * @return true if <b>sym</b> has exteranl linkage
    */
    private static boolean hasExternalLinkage(Symbol sym) {
        // A Symbol has external linkage if
        // 1. Symbol has "extern" specifier or
        // 2. parent SymbolTable is of Type TranslationUnit && and not "static"
        try {
            if (SymbolTools.containsSpecifier(sym, Specifier.EXTERN)) {
                return true;
            }
            if (SymbolTools.containsSpecifier(sym, Specifier.STATIC)) {
                return false;
            }
            // getTypeSpecifiers() is not appropriate for this method since this
            // method is called extensibly and getTypeSpecifiers() consumes
            // extra memory for the returned list.
/*
            for(Specifier spec1 : (List<Specifier>)sym.getTypeSpecifiers()) {
                if(spec1 == Specifier.EXTERN)
                    return true;
                if(spec1 == Specifier.STATIC)
                    return false;
            }
*/
        } catch(Exception e) {
            PrintTools.printlnStatus("Fatal Error: Non Specifier List", 0);
            e.printStackTrace();
            System.exit(1);
        }
        Declaration decl = sym.getDeclaration();
        if (decl == null) {
            return false;
        }
        // check if parent SymbolTable is a TranslationUnit
        SymbolTable parent_sym =
                IRTools.getAncestorOfType(decl, SymbolTable.class);
        if (parent_sym instanceof TranslationUnit) {
            return true;
        }
        return false;
    }

    /**
    * Checks if the given Object <b>obj</b> refers to the same Symbol as this
    * Symbol.
    * @param obj
    * @return true if <b>obj</b> refers to the same memory location as this
    * Symbol.
    */
    @Override
    public boolean equals(Object obj) {
        return isSameSymbol(obj);
    }

    /**
    * If there are multiple Symbols with external linkage and same name,
    * hashCode() returns same value for those symbols
    * @return the computed hash code.
    */
    @Override
    public int hashCode() {
        // If Symbol has external linkage
        // return hashcode for symbol name
        if (hasExternalLinkage(this)) {
            return getSymbolName().hashCode();
        }
        // If Symbol has internal linkage
        // return default hashcode
        return super.hashCode();
    }

}
