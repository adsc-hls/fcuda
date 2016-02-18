package cetus.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
* Represents the part of a declaration that is the name of the symbol, some type
* information, and initial values. This class actually is more similar to what
* the C++ grammar calls an init-declarator.  Many different constructors are
* provided because Java does not have default arguments.
*/
public abstract class Declarator implements Cloneable, Traversable {

    /** The default print method */
    private static Method class_print_method;

    /** The print method for the declarator */
    protected Method object_print_method;

    static {
        Class<?>[] parameters = new Class<?>[2];
        try {
            parameters[0] = Declarator.class;
            parameters[1] = PrintWriter.class;
            class_print_method =
                    parameters[0].getMethod("defaultPrint", parameters);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** The parent object */
    protected Traversable parent;

    /** The list of child objects */
    protected List<Traversable> children;

    /** The list of specifiers that appear before the declarator ID */
    protected List<Specifier> leading_specs;

    /** The list of specifiers that appear after the declarator ID */
    protected List<Specifier> trailing_specs;

    /** Exception specification - not used in C */
    private ExceptionSpecification espec;

    protected static final List empty_list =
            Collections.unmodifiableList(new ArrayList<Object>(0));

    /**
    * Constructs an empty declarator.
    */
    protected Declarator() {
        object_print_method = class_print_method;
        parent = null;
        children = new ArrayList<Traversable>(1);
        leading_specs = null;
        trailing_specs = null;
    }

    /**
    * Constructs a declarator with the given size of the child list.
    * This method is identical to {@link #Declarator()}.
    */
    protected Declarator(int size) {
        object_print_method = class_print_method;
        parent = null;
        children = new ArrayList<Traversable>(size);
        leading_specs = null;
        trailing_specs = null;
    }

    /**
    * Inserts a new child declaration to the parameter list.
    *
    * @param decl the new parameter declaration to be inserted.
    * @throws NotAnOrphanException if <b>decl</b> has a parent object.
    */
    public abstract void addParameter(Declaration decl);

    /**
    * Inserts a new child declaration to the parameter list before the given
    * reference declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be inserted.
    * @throws IllegalArgumentException if <b>ref</b> is not found.
    * @throws NotAnOrphanException if <b>decl</b> has a parent object.
    */
    public abstract void addParameterBefore(Declaration ref, Declaration decl);

    /**
    * Inserts a new child declaration to the parameter list after the given
    * reference declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be inserted.
    * @throws IllegalArgumentException if <b>ref</b> is not found.
    * @throws NotAnOrphanException if <b>decl</b> has a parent object.
    */
    public abstract void addParameterAfter(Declaration ref, Declaration decl);


  /* **AP**
    Adding method for removing parameters from ProcedureDeclarator list
  */
  public abstract void removeParameter(Declaration decl);
//   {
//     int index = Tools.indexByReference(parameters, decl);
// 
//     if (index == -1)
//       throw new IllegalArgumentException();
// 
//     decl.setParent(null);  
//     parameters.remove(decl);
//   }


    /**
    * Appends a new specifier to the list of trailing specifiers.
    *
    * @param spec the new specifier to be appended.
    */
    public void addTrailingSpecifier(Specifier spec) {
        if (trailing_specs == null) {
            trailing_specs = new ArrayList<Specifier>(2);
        }
        trailing_specs.add(spec);
    }

    @Override
    public Declarator clone() {
        Declarator d = null;
        try {
            d = (Declarator)super.clone();
        } catch(CloneNotSupportedException e) {
            throw new InternalError();
        }
        d.parent = null;
        d.children = new ArrayList<Traversable>(children.size());
        if (leading_specs != null) {
            d.leading_specs = new ArrayList<Specifier>(leading_specs);
        } else {
            d.leading_specs = null;
        }
        if (trailing_specs != null) {
            d.trailing_specs = new ArrayList<Specifier>(trailing_specs);
        } else {
            d.trailing_specs = null;
        }
        d.espec = espec;
        return d;
    }

    /**
    * Checks if the given object is equal to the declarator.
    *
    * @param o the object to be compared.
    * @return true if {@code o == this}.
    */
    @Override
    public boolean equals(Object o) {
        return (o == this);
    }

    /**
    * Returns the hash code of the declarator.
    *
    * @return the identity hash code of the declarator.
    */
    @Override public int hashCode() {
        return System.identityHashCode(this);
    }

    /**
    * Prints a declarator to a stream.
    *
    * @param d The declarator to print.
    * @param o The writer on which to print the declarator.
    */
    public static void defaultPrint(Declarator d, PrintWriter o) {
        PrintTools.printList(d.leading_specs, o);
        PrintTools.printList(d.trailing_specs, o);
        if (d.getInitializer() != null) {
            d.getInitializer().print(o);
        }
    }

    /**
    * Converts the declarator to a string by calling the default print method.
    * All sub classes will be using this method unless special handling is
    * necessary.
    */
    @Override
    public String toString() {
        StringWriter sw = new StringWriter(80);
        print(new PrintWriter(sw));
        return sw.toString();
    }

    /* Traversable interface */
    public List<Traversable> getChildren() {
        return children;
    }

    /**
    * Returns the variable initializer of the declarator if one exists.
    *
    * @return the initializer or null.
    */
    public Initializer getInitializer() {
        if (children.size() > 0) {
            return (Initializer)children.get(0);
        } else {
            return null;
        }
    }

    /**
    * Returns a List of Function Parameter
    *
    * @return List null is returned when there is no Function Parameter in the
    * Declarator
    */
    public abstract List<Declaration> getParameters();

    /**
    * Returns the parameter at specified index.
    * @param index - zero-based index of the required parameter
    * @return - parameter at specified index or null
    */
    public Declaration getParameter(int index) {
        List<Declaration> params = getParameters();
        if (params != null && index >= 0 && index < params.size()) {
            return params.get(index);
        }
        return null;
    }

    /* Traversable interface */
    public Traversable getParent() {
        return parent;
    }

    /**
    * Returns the list of specifiers trailing the declarator.
    *
    * @return the trailing specifiers.
    */
    public List<Specifier> getArraySpecifiers() {
        return trailing_specs;
    }

    /**
    * Returns the list of specifiers leading the declarator.
    *
    * @return the leading specifiers.
    */
    public List<Specifier> getSpecifiers() {
        return leading_specs;
    }

    /**
    * Returns the symbol declared by this declarator.
    *
    * @return the name ID of the declarator.
    */
    public abstract IDExpression getID();

    /** Prints the declarator to the specified print writer. */
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

    /**
    * This traversable interface is not supported for declarators.
    *
    * @throws UnsupportedOperationException always.
    */
    public void removeChild(Traversable child) {
        throw new UnsupportedOperationException();
    }

    /**
    * This traversable interface is not supported for declarators.
    *
    * @throws UnsupportedOperationException always.
    */
    public void setChild(int index, Traversable t) {
        throw new UnsupportedOperationException();
    }

    /** Returns the name ID of the declarator */
    protected abstract IDExpression getDirectDeclarator();

    /** Sets the name ID with the given new ID <b>direct_decl</b> */
    protected abstract void setDirectDeclarator(IDExpression direct_decl);

    /**
    * Sets the initial value of the variable.  The initial value cannot be set
    * in the constructor, for the purpose of limiting the number of
    * constructors.
    *
    * @param init An initial value for the variable.
    */
    public void setInitializer(Initializer init) {
        if (getInitializer() != null) {
            getInitializer().setParent(null);
            if (init != null) {
                children.set(0, init);
                init.setParent(this);
            } else {
                children.clear();
            }
        } else {
            if (init != null) {
                children.add(init);
                init.setParent(this);
            } else {
                children.clear();
            }
        }
    }

    /* Traversable interface */
    public void setParent(Traversable t) {
        parent = t;
    }

}
