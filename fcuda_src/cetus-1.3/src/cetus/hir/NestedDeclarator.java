package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.List;

/**
* Represents a nested declarator that may contain another declarator within
* itself. e.g., function pointers and pointers to a chunck of arrays.
*/
public class NestedDeclarator extends Declarator implements Symbol {

    /** Default class print method. */
    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = NestedDeclarator.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** Flag for identifying a nested declarator for function pointers. */
    private boolean has_param = false;

    /** Common initialization process. */
    private void initialize(Declarator nested_decl, List params) {
        object_print_method = class_print_method;
        if (nested_decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(nested_decl);
        nested_decl.setParent(this);
        if (params != null) {
            has_param = true;
            for (int i = 0; i < params.size(); i++) {
                Traversable decl = (Traversable)params.get(i);
                if (decl.getParent() != null) {
                    throw new NotAnOrphanException();
                }
                children.add(decl);
                decl.setParent(this);
            }
        }
    }

    /**
    * Constructs a new nested declarator with the given child declarator.
    *
    * @param nested_decl the child declarator to be added.
    */
    public NestedDeclarator(Declarator nested_decl) {
        super(1);
        initialize(nested_decl, null);
        leading_specs = new ArrayList<Specifier>(1);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given child declarator and the
    * list of parameters.
    *
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    */
    public NestedDeclarator(Declarator nested_decl, List params) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        leading_specs = new ArrayList<Specifier>(1);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given list of leading
    * specifiers, the child declarator, and the list of parameters.
    *
    * @param leading_specs the list of leading specifiers.
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    */
    @SuppressWarnings("unchecked")
    public NestedDeclarator(List leading_specs, Declarator nested_decl,
                            List params) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given leading specifier, the
    * child declarator, and the list of parameters.
    *
    * @param spec the leading specifier.
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    */
    public NestedDeclarator(Specifier spec, Declarator nested_decl,
                            List params) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        leading_specs = new ArrayList<Specifier>(1);
        leading_specs.add(spec);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given list of leading
    * specifers, the child declarator, the list of parameters, and the list of
    * trailing specifiers.
    *
    * @param leading_specs the list of leading specifiers.
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    * @param trailing_specs the list of trailing specifiers.
    */
    @SuppressWarnings("unchecked")
    public NestedDeclarator(List leading_specs, Declarator nested_decl,
                            List params, List trailing_specs) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(trailing_specs);
    }

    /**
    * Inserts a new parameter declaration to the nested declarator.
    *
    * @param decl the new parameter declaration to be added.
    */
    @Override
    public void addParameter(Declaration decl) {
        has_param = true;
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (getInitializer() == null) { // initializer is positioned at the end.
            children.add(decl);
        } else {
            children.add(children.size() - 1, decl);
        }
        decl.setParent(this);
    }

    /**
    * Inserts a new parameter declaration to the nested declarator before the
    * reference parameter declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    @Override
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
    * Inserts a new parameter declaration to the nested declarator after the
    * reference parameter declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    @Override
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
    throw new UnsupportedOperationException();
  }

    /**
    * Returns a clone of the nested declarator.
    */
    @Override
    public NestedDeclarator clone() {
        NestedDeclarator d = (NestedDeclarator)super.clone();
        Declarator id = getDeclarator().clone();
        d.children.add(id);
        id.setParent(d);
        if (children.size() > 1) {
            for (int i = 1; i < children.size(); i++) {
                Traversable child = children.get(i);
                if (child instanceof Declaration) {
                    Declaration decl = ((Declaration)child).clone();
                    d.children.add(decl);
                    decl.setParent(d);
                // was getting ClassCastException on statements like:
                // int (*abc)[2] = temp;
                // (where: int temp[][2] = {{2,3},{4,5}};)
                } else if (child instanceof Initializer) {
                    Initializer init = ((Initializer)child).clone();
                    d.children.add(init);
                    init.setParent(d);
                }
            }
        }
        d.has_param = has_param;
        return d;
    }

    /**
    * Prints a nested declarator to a stream.
    *
    * @param d The declarator to print.
    * @param o The writer on which to print the declarator.
    */
    public static void defaultPrint(NestedDeclarator d, PrintWriter o) {
        PrintTools.printListWithSpace(d.leading_specs, o);
        o.print("(");
        d.getDeclarator().print(o);
        o.print(")");
        if (d.has_param) {
            o.print("(");
            int num_param = d.children.size() - 1;
            if (d.getInitializer() != null) {
                num_param--;
            }
            if (num_param > 0) {
                PrintTools.printListWithComma(
                        d.children.subList(1, num_param + 1), o);
            }
            o.print(")");
        }
        PrintTools.printListWithSpace(d.trailing_specs, o);
        if (d.getInitializer() != null) {
            d.getInitializer().print(o);
        }
    }

    /**
    * Returns the child declarator of the nested declarator.
    */
    public Declarator getDeclarator() {
        return (Declarator)children.get(0);
    }

    /**
    * Returns the list of parameters if the declarator represents a function.
    * @return the list of parameters if it does, {@code null} otherwise.
    */
    @Override
    public List<Declaration> getParameters() {
        if (!has_param) {
            return null;
        }
        List<Declaration> ret = new ArrayList<Declaration>(children.size() - 1);
        for (int i = 1; i < children.size(); i++) {
            ret.add((Declaration)children.get(i));
        }
        ret.remove(getInitializer());
        return ret;
    }

    /**
    * Returns the name ID declared by the nested declarator.
    */
    public IDExpression getID() {
        return getDeclarator().getDirectDeclarator();
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
    * Returns the initializer of the nested declarator.
    *
    * @return the initializer if exists, null otherwise.
    */
    public Initializer getInitializer() {
        if (!children.isEmpty()) {
            if (children.get(children.size() - 1) instanceof Initializer) {
                return (Initializer)children.get(children.size() - 1);
            }
        }
        return null;
    }

    /**
    * Assigns a new initializer for the nested declarator. The existing
    * initializer is dicarded.
    */
    public void setInitializer(Initializer init) {
        if (getInitializer() != null) {
            getInitializer().setParent(null);
            if (init != null) {
                children.set(children.size() - 1, init);
                init.setParent(this);
            }
        } else {
            if (init != null) {
                children.add(init);
                init.setParent(this);
            }
        }
    }

    /**
    * Checks if the nested declarator is used to represent a procedure call.
    */
    public boolean isProcedure() {
        return has_param;
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
    public String getSymbolName() {
        return getID().toString();
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getArraySpecifiers() {
        return trailing_specs;
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

    protected IDExpression getDirectDeclarator() {
        return getDeclarator().getDirectDeclarator();
    }

    protected void setDirectDeclarator(IDExpression direct_decl) {
        getDeclarator().setDirectDeclarator(direct_decl);
    }

}
