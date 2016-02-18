package cetus.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

/**
* Base class for all expressions. Expressions are compared lexically against
* other expressions when used in collections.
*/
public abstract class Expression
                implements Cloneable, Comparable<Expression>, Traversable {

    /** The print method for the expression */
    protected Method object_print_method;

    /** The parent object of the expression */
    protected Traversable parent;

    /** All children must be Expressions. (Exception: statement expression) */
    protected List<Traversable> children;

    /**
    * Determines whether this expression should have a set of parentheses
    * around it when printed.
    */
    protected boolean needs_parens;

    /** Empty child list for expressions having no children */
    protected static final List empty_list =
            Collections.unmodifiableList(new ArrayList<Object>(0));

    /** Constructor for derived classes. */
    protected Expression() {
        parent = null;
        children = new ArrayList<Traversable>(1);
        needs_parens = true;
    }

    /**
    * Constructor for derived classes.
    *
    * @param size The initial size for the child list.
    */
    @SuppressWarnings("unchecked")
    protected Expression(int size) {
        parent = null;
        if (size < 0) {
            children = empty_list;
        } else {
            children = new ArrayList<Traversable>(size);
        }
        needs_parens = true;
    }

    /**
    * Creates and returns a deep copy of this expression.
    *
    * @return a deep copy of this expression.
    */
    @Override
    public Expression clone() {
        Expression o = null;
        try {
            o = (Expression)super.clone();
        } catch(CloneNotSupportedException e) {
            throw new InternalError();
        }
        o.object_print_method = object_print_method;
        o.parent = null;
        if (children != null) {
            o.children = new ArrayList<Traversable>(children.size());
            for (int i = 0; i < children.size(); i++) {
                Traversable new_child = children.get(i);
                if (new_child instanceof Expression) {
                    new_child = ((Expression)new_child).clone();
                // Handling of StatementExpression (gcc extension)
                } else if (new_child instanceof Statement) {
                    new_child = ((Statement)new_child).clone();
                }
                new_child.setParent(o);
                o.children.add(new_child);
            }
        } else {
            o.children = null;
        }
        o.needs_parens = needs_parens;
        return o;
    }

    /* Comparable interface */
    public int compareTo(Expression e) {
        if (equals(e)) {
            return 0;
        } else {
            return toString().compareTo(e.toString());
        }
    }

    /**
    * Checks if the given object is has the same type with this expression and
    * its children is same with this expression's. The sub classes of expression
    * should call this method first and proceed with more checking if they
    * have additional fields to be checked.
    * @param o the object to be compared with.
    * @return true if {@code o!=null}, {@code this.getClass()==o.getClass()},
    *       and {@code this.children.equals(o.children) ||
    *       this.children==o.children==null}
    */
    @Override
    public boolean equals(Object o) {
        if (o == null || this.getClass() != o.getClass()) {
            return false;
        }
        if (children == null) {
            return (((Expression)o).children == null);
        } else {
            return children.equals(((Expression)o).children);
        }
    }

    /**
    * Returns the hash code of the expression. It returns the hash code of the
    * string representation since expressions are compared lexically.
    *
    * @return the integer hash code of the expression.
    */
    @Override
    public int hashCode() {
        return hashCode(0);
    }

    /**
    * Returns cumulative hash code with the given initial value, {@code h}.
    * This is intended for optimizing memory usage when computing hash code of
    * an expression object. Previous approach constructs a string representation
    * of the expression to get the hash code while this approach uses the same
    * algorithm as {@link String#hashCode()} without creating a string. All sub
    * classes of {@code Expression} have there specific implementations of this
    * method and {@link #hashCode()} simply returns {@code hashCode(0)}.
    */
    // NOTE for developers: Make this method consistent with toString() for all
    // sub classes of expression.
    protected int hashCode(int h) {
        return hashCode(this, h);
    }

    /**
    * Returns a list of subexpressions of this expression that match
    * <var>expr</var> using its equals method.
    *
    * @param expr The subexpression sought.
    * @return a list of matching subexpressions, which may be empty.
    */
    public List<Expression> findExpression(Expression expr) {
        List<Expression> result = new LinkedList<Expression> ();
        if (expr != null) {
            BreadthFirstIterator iter = new BreadthFirstIterator(this);
            while (iter.hasNext()) {
                Object obj = iter.next();
                if (expr.equals(obj)) {
                    result.add((Expression)obj);
                }
            }
        }
        return result;
    }

    /* Traversable interface */
    public List<Traversable> getChildren() {
        return children;
    }

    /* Traversable interface */
    public Traversable getParent() {
        return parent;
    }

    /**
    * Get the parent Statement containing this Expression.
    *
    * @return the enclosing Statement or null if this Expression
    *   is not inside a Statement.
    */
    public Statement getStatement() {
        Traversable t = this;
        do {
            t = t.getParent();
        } while (t != null && !(t instanceof Statement));
        return (Statement)t;
    }

    /**
    * Prints the expression on the specified print writer.
    *
    * @param o the target print writer.
    */
    public void print(PrintWriter o) {
        if (object_print_method == null) {
            return;
        }
        try {
            object_print_method.invoke(null, new Object[] {this, o});
        } catch(IllegalAccessException e) {
            System.err.println(e);
            e.printStackTrace();
            Tools.exit(1);
        } catch(InvocationTargetException e) {
            System.err.println(e.getCause());
            e.printStackTrace();
            Tools.exit(1);
        }
    }

    // TODO: what is this method intended for ?
    // TODO: remove if not used.
    public void printSelf() {
        /*
           System.out.print("Source form: ");
           print(System.out);
           System.out.println("");
         */
        if (this instanceof BinaryExpression) {
            System.out.print(((BinaryExpression) this).getOperator());
        } else {
            System.out.print(getClass().getName().substring(10));
        }
        if (this.parent instanceof Expression) {
            System.out.print(" " + "@" + hashCode());
        }
        /*
           Fix Me
           Not all children are Expression
         */
        Iterator iter = children.iterator();
        Expression e = null;
        while (iter.hasNext()) {
            e = (Expression) iter.next();
            if (e.getChildren().size() != 0) {
                System.out.print(" " + "@" + e.hashCode());
            } else {
                System.out.print(" ");
                e.print(new PrintWriter(System.out));
            }
        }
    }

    /**
    * This operation is not allowed.
    * @throws UnsupportedOperationException always
    */
    public void removeChild(Traversable child) {
        throw new UnsupportedOperationException(
                "Expressions do not support removal of arbitrary children.");
    }

    /**
    * @throws NotAnOrphanException if <b>t</b> has a parent object.
    * @throws IllegalArgumentException if <b>index</b> is out-of-range or
    * <b>t</b> is not an expression.
    */
    public void setChild(int index, Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (!(t instanceof Expression) || index >= children.size()) {
            throw new IllegalArgumentException();
        }
        // Detach the old child
        if (children.get(index) != null) {
            children.get(index).setParent(null);
        }
        children.set(index, t);
        t.setParent(this);
    }

    /**
    * Sets whether the expression needs to have
    * an outer set of parentheses printed around it.
    *
    * @param f True to use parens, false to not use parens.
    */
    public void setParens(boolean f) {
        needs_parens = f;
    }

    /**
    * Checks if the expression needs parentheses around itself when printed.
    */
    public boolean needsParens() {
        return needs_parens;
    }

    /* Traversable interface */
    public void setParent(Traversable t) {
        // expressions can appear in many places so it's probably not
        // worth it to try and provide instanceof checks against t here
        parent = t;
    }

    /**
    * Overrides the print method for this object only.
    *
    * @param m The new print method.
    */
    public void setPrintMethod(Method m) {
        object_print_method = m;
    }

    /**
    * Swaps two expression on the IR tree.  If neither this expression nor
    * <var>expr</var> has a parent, then this function has no effect. Otherwise,
    * each expression ends up with the other's parent and exchange positions in
    * the parents' lists of children.
    *
    * @param expr The expression with which to swap this expression.
    * @throws IllegalArgumentException if <var>expr</var> is null.
    * @throws IllegalStateException if the types of the expressions
    *   are such that they would create inconsistent IR when swapped.
    */
    public void swapWith(Expression expr) {
        if (expr == null) {
            throw new IllegalArgumentException();
        }
        if (this == expr) {
            // swap with self does nothing
            return;
        }
        // The rest of this must be done in a very particular order.
        // Be EXTREMELY careful changing it.
        Traversable this_parent = this.parent;
        Traversable expr_parent = expr.parent;
        int this_index = -1, expr_index = -1;
        if (this_parent != null) {
            this_index = Tools.identityIndexOf(this_parent.getChildren(), this);
            if (this_index == -1) {
                throw new IllegalStateException();
            }
        }
        if (expr_parent != null) {
            expr_index = Tools.identityIndexOf(expr_parent.getChildren(), expr);
            if (expr_index == -1) {
                throw new IllegalStateException();
            }
        }
        // detach both so setChild won't complain
        expr.parent = null;
        this.parent = null;
        if (this_parent != null) {
            this_parent.getChildren().set(this_index, expr);
            expr.setParent(this_parent);
        }
        if (expr_parent != null) {
            expr_parent.getChildren().set(expr_index, this);
            this.setParent(expr_parent);
        }
    }

    /** Returns a string representation of the expression */
    @Override
    public String toString() {
        StringWriter sw = new StringWriter(40);
        print(new PrintWriter(sw));
        return sw.toString();
    }

    /**
    * Verifies three properties of this object:
    * (1) All children are not null, (2) the parent object has this
    * object as a child, (3) all children have this object as the parent.
    *
    * @throws IllegalStateException if any of the properties are not true.
    */
    public void verify() throws IllegalStateException {
        if (parent != null && !parent.getChildren().contains(this)) {
            throw new IllegalStateException(
                    "parent does not think this is a child");
        }
        if (children != null) {
            if (children.contains(null)) {
                throw new IllegalStateException("a child is null");
            }
            for (Traversable t : children) {
                if (t.getParent() != this) {
                    throw new IllegalStateException(
                            "a child does not think this is the parent");
                }
            }
        }
    }

    /**
    * Common operation used in constructors - adds the specified traversable
    * object at the end of the child list.
    *
    * @param t the new child object to be added.
    * @throws NotAnOrphanException
    */
    protected void addChild(Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(t);
        t.setParent(this);
    }

    protected static int hashCode(Object o, int h) {
        String s = o.toString();
        for (int i = 0; i < s.length(); i++) {
            h = 31 * h + s.charAt(i);
        }
        return h;
    }

    protected static int hashCode(List list, String sep, int h) {
        if (list.size() > 0) {
            Object o = list.get(0);
            if (o instanceof Expression) {
                h = ((Expression)o).hashCode(h);
            } else {
                h = hashCode(o, h);
            }
            for (int i = 1; i < list.size(); i++) {
                for (int j = 0; j < sep.length(); j++) {
                    h = 31 * h + sep.charAt(j);
                }
                o = list.get(i);
                if (o instanceof Expression) {
                    h = ((Expression)o).hashCode(h);
                } else {
                    h = hashCode(o, h);
                }
            }
        }
        return h;
    }

}
