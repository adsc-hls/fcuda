package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/** Represents a declarator for a Procedure in a VariableDeclaration. */
public class ProcedureDeclarator extends Declarator implements Symbol {

    /** Default method for printing procedure declarator */
    private static Method class_print_method;

    /** Default print method assignment */
    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = ProcedureDeclarator.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** Not used in C */
    private ExceptionSpecification espec;

    /** Common initialization process with the given input */
    private void initialize(IDExpression direct_decl, List params) {
        object_print_method = class_print_method;
        if (direct_decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        // Forces use of NameID.
        if (!(direct_decl instanceof NameID)) {
            direct_decl = new NameID(direct_decl.toString());
        }
        children.add(direct_decl);
        direct_decl.setParent(this);
        for (int i = 0; i < params.size(); i++) {
            Traversable decl = (Traversable)params.get(i);
            if (decl.getParent() != null) {
                throw new NotAnOrphanException();
            }
            children.add(decl);
            decl.setParent(this);
        }
    }

    /** 
    * Constructs a new procedure declarator with the given ID and the list of
    * parameters.
    * 
    * @param direct_decl the IDExpression used for this procedure name; it is
    * highly recommended to use {@link NameID} since this constructor internally
    * replaces the parameter with an equivalent <b>NameID</b> object.
    * @param params the list of function parameters.
    */
    public ProcedureDeclarator(IDExpression direct_decl, List params) {
        super(1 + params.size());
        initialize(direct_decl, params);
        leading_specs = new ArrayList<Specifier>(1);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Consturcts a new procedure declarator with the given ID, list of
    * parameters, and the trailing specifiers.
    *
    * @param direct_decl the IDExpression used for this procedure name; it is
    * highly recommended to use {@link NameID} since this constructor internally
    * replaces the parameter with an equivalent <b>NameID</b> object.
    * @param params the list of function parameters.
    * @param trailing_specs the list of trailing specifiers.
    */
    @SuppressWarnings("unchecked")
    public ProcedureDeclarator(IDExpression direct_decl, List params,
                               List trailing_specs) {
        super(1 + params.size());
        initialize(direct_decl, params);
        this.leading_specs = new ArrayList<Specifier>(1);
        this.trailing_specs = new ArrayList<Specifier>(trailing_specs);
    }

    /**
    * Constructs a new procedure declarator with the given leading specifiers,
    * the ID, and the list of parameters. This is the most commonly used
    * constructor for C input language.
    *
    * @param leading_specs the list of leading specifiers.
    * @param direct_decl the IDExpression used for this procedure name; it is
    * highly recommended to use {@link NameID} since this constructor internally
    * replaces the parameter with an equivalent <b>NameID</b> object.
    * @param params the list of function parameters.
    */
    @SuppressWarnings("unchecked")
    public ProcedureDeclarator(List leading_specs,
                               IDExpression direct_decl, List params) {
        super(1 + params.size());
        initialize(direct_decl, params);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new procedure declarator with the given leading specifiers,
    * the ID, the trailing specifiers, and the exception specification. This
    * constructor is not used for C programs.
    *
    * @param leading_specs the list of leading specifiers.
    * @param direct_decl the IDExpression used for this procedure name; it is
    * highly recommended to use {@link NameID} since this constructor internally
    * replaces the parameter with an equivalent <b>NameID</b> object.
    * @param params the list of function parameters.
    * @param trailing_specs the list of trailing specifiers.
    * @param espec the exception specification.
    */
    @SuppressWarnings("unchecked")
    public ProcedureDeclarator(List leading_specs,
                               IDExpression direct_decl, List params,
                               List trailing_specs,
                               ExceptionSpecification espec) {
        super(1 + params.size());
        initialize(direct_decl, params);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(trailing_specs);
        this.espec = espec;
    }

    /**
    * Inserts a new parameter declaration at the end of the parameter list.
    * 
    * @param decl the new parameter declaration to be added.
    */
    public void addParameter(Declaration decl) {
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(decl);
        decl.setParent(this);
    }

    /**
    * Inserts a new parameter declaration before the specified reference
    * parameter declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    public void addParameterBefore(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(index, decl);
        decl.setParent(this);
    }

    /**
    * Inserts a new parameter declaration after the specified reference
    * parameter declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    public void addParameterAfter(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(index + 1, decl);
        decl.setParent(this);
    }

  /* *AP*
    Adding method for removing parameters from declarator list
  */
  public void removeParameter(Declaration decl)
  {
    int index = Tools.indexByReference(children, decl);

    if (index == -1)
      throw new IllegalArgumentException();

    children.remove(index);
    decl.setParent(null);
  }

    /** Returns a clone of the procedure declarator. */
    @Override
    public ProcedureDeclarator clone() {
        ProcedureDeclarator d = (ProcedureDeclarator)super.clone();
        IDExpression id = getDirectDeclarator().clone();
        d.children.add(id);
        id.setParent(d);
        if (children.size() > 1) {
            for (int i = 1; i < children.size(); i++) {
                Declaration decl = ((Declaration)children.get(i)).clone();
                d.children.add(decl);
                decl.setParent(d);
            }
        }
        d.espec = espec;
        return d;
    }

    /**
    * Prints a procedure declarator to a stream.
    *
    * @param d The declarator to print.
    * @param o The writer on which to print the declarator.
    */
    public static void defaultPrint(ProcedureDeclarator d, PrintWriter o) {
        PrintTools.printList(d.leading_specs, o);
        d.getDirectDeclarator().print(o);
        o.print("(");
        if (d.children.size() > 1) {
            PrintTools.printListWithComma(
                    d.children.subList(1, d.children.size()), o);
        }
        o.print(")");
        PrintTools.printListWithSeparator(d.trailing_specs, o, " ");
    }

    /** Returns the name ID of the procedure declarator. */
    protected IDExpression getDirectDeclarator() {
        return (IDExpression)children.get(0);
    }

    /** Returns the name ID of the procedure declarator. */
    public IDExpression getID() {
        return getDirectDeclarator();
    }

    /**
    * Returns the list of parameter declaration of the procedure declarator.
    */
    public List<Declaration> getParameters() {
        List<Declaration> ret = new ArrayList<Declaration>(children.size() - 1);
        for (int i = 1; i < children.size(); i++) {
            ret.add((Declaration) children.get(i));
        }
        return ret;
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

    /* Symbol interface */
    public String getSymbolName() {
        return getDirectDeclarator().toString();
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getTypeSpecifiers() {
        Traversable t = this;
        while (!(t instanceof Declaration)) {
            t = t.getParent();
        }
        List ret = new ArrayList(4);
        if (t instanceof VariableDeclaration) {
            ret.addAll(((VariableDeclaration)t).getSpecifiers());
        } else {
            return null;
        }
        ret.addAll(leading_specs);
        return ret;
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getArraySpecifiers() {
        return null;
    }

    /** Sets the direct declarator with the specified new ID */
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

    /**
    * Returns the parent declaration of the procedure declarator if one exists.
    * The procedure declarators that are not included in the IR tree
    * (e.g., field of a procedure object) does not have any parent declaration
    * since search is not posssible (hence null is returned), whereas a
    * procedure declarator that appears as a child of a variable declaration has
    * a specific parent declaration.
    *
    * @return the parent declaration if one exists, null otherwise. 
    */
    public Declaration getDeclaration() {
        return IRTools.getAncestorOfType(this, Declaration.class);
    }

}
