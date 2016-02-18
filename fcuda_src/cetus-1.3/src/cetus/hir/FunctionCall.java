package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
* Represents a function or method call.
*/
public class FunctionCall extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = FunctionCall.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Creates a function call having no arguments.
    *
    * @param function An expression that evaluates to a function.
    * @param args Arguments
    * @throws IllegalArgumentException if <b>function</b> is invalid.
    * @throws NotAnOrphanException if <b>function</b> has a parent object.
    */
    public FunctionCall(Expression function, Expression ... args) {
        super(args.length + 1);
        object_print_method = class_print_method;
        needs_parens = false;
        addChild(function);
        for (Expression arg : args) {
            addChild(arg);
        }
    }

    /**
    * Creates a function call having a list of arguments.
    *
    * @param function An expression that evaluates to a function.
    * @param args A list of arguments to the function.
    * @throws IllegalArgumentException if <b>function</b> or an element of
    * <b>args</b> is invalid.
    * @throws NotAnOrphanException if <b>function</b> or an element of
    * <b>args</b> has a parent object.
    */
    public FunctionCall(Expression function, List args) {
        super(args.size() + 1);
        object_print_method = class_print_method;
        needs_parens = false;
        addChild(function);
        for (Object arg : args) {
            addChild((Traversable)arg);
        }
    }

    /**
    * Inserts the specified expression at the end of the argument list.
    *
    * @param expr the new argument to be inserted.
    *
    * @throws IllegalArgumentException if <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void addArgument(Expression expr) {
        addChild(expr);
    }

    /**
    * Prints a function call to a stream.
    *
    * @param c The call to print.
    * @param o The writer on which to print the call.
    */
    public static void defaultPrint(FunctionCall c, PrintWriter o) {
        if (c.needs_parens) {
            o.print("(");
        }
        o.print(c.getName());
        o.print("(");
        PrintTools.printListWithComma(c.children.
                                      subList(1, c.children.size()), o);
        o.print(")");
        if (c.needs_parens) {
            o.print(")");
        }
    }

    /**
    * Returns a string representation of the function call. The returned
    * string is equivalent to the printed stream through {@link #defaultPrint}.
    * @return the string representation.
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        if (needs_parens) {
            sb.append("(");
        }
        sb.append(getName());
        sb.append("(");
        sb.append(PrintTools.listToString(
                children.subList(1, children.size()), ", "));
        sb.append(")");
        if (needs_parens) {
            sb.append(")");
        }
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        if (needs_parens) {
            h = 31 * h + '(';
        }
        h = getName().hashCode(h);
        h = 31 * h + '(';
        h = hashCode(children.subList(1, children.size()), ", ", h);
        h = 31 * h + ')';
        if (needs_parens) {
            h = 31 * h + ')';
        }
        return h;
    }

    /**
    * Returns the specified argument
    * @param n - index of required argument (first argument has index zero)
    * @return the argument at the specified position.
    * @throws IllegalArgumentException if <b>n</b> is out of range.
    */
    public Expression getArgument(int n) {
        if (n < 0 || n >= children.size() - 1) {
            throw new IllegalArgumentException();
        }
        return (Expression)children.get(n + 1);
    }

    /**
    * Returns the list of arguments of the function call.
    *
    * @return the argument list.
    */
    public List<Expression> getArguments() {
        List<Expression> ret = new ArrayList<Expression>(children.size() - 1);
        for (int i = 1; i < children.size(); i++) {
            ret.add((Expression)children.get(i));
        }
        return ret;
    }

    /**
    * Returns the name of the function call.
    *
    * @return the expression used as the function name.
    */
    public Expression getName() {
        return (Expression)children.get(0);
    }

    /**
    * Returns the number of arguments being passed to this function call.
    *
    * @return the number of arguments.
    */
    public int getNumArguments() {
        return children.size() - 1;
    }

    /**
    * Returns the Procedure object for the function that is being called.
    *
    * @return the Procedure object if found, null otherwise.
    */
    public Procedure getProcedure() {
        Traversable t = this;
        do {
            if (t == null) {
                return null;
            }
            t = t.getParent();
        } while (!(t instanceof SymbolTable));
        SymbolTable table = (SymbolTable)t;
        Declaration decl = null;
        try {
            decl = table.findSymbol((IDExpression)getName());
        } catch(ClassCastException e) {
            return null;
        }
        if (decl instanceof Procedure) {
            return (Procedure)decl;
        } else if (decl instanceof VariableDeclaration) {
            t = this;
            while (t.getParent() != null) {
                t = t.getParent();
            }
            Program prog = (Program)t;
            List<Traversable> prog_children = prog.getChildren();
            for (int i = 0; i < prog_children.size(); i++) {
                TranslationUnit tu = (TranslationUnit)prog_children.get(i);
                Object o = tu.findSymbol((IDExpression)getName());
                if (o != null && o instanceof Procedure) {
                    return (Procedure)o;
                }
            }
            return null;
        } else {
            return null;
        }
    }

    /**
    * Returns the list of specifiers that define the return type of the function
    * call.
    *
    * @return the specifier list.
    */
    @SuppressWarnings("unchecked")
    public List getReturnType() {
        Traversable t = this;
        do {
            t = t.getParent();
        } while (!(t instanceof SymbolTable));
        SymbolTable table = (SymbolTable)t;
        Declaration decl = null;
        try {
            decl = table.findSymbol((IDExpression)getName());
        } catch(ClassCastException e) {
            System.err.println(
                    "Cast Exception in FunctionCall.getReturnType: "+getName());
        }
        if (decl == null) {
            return empty_list;
        } else if (decl instanceof VariableDeclaration) {
            List list = new ArrayList(
                    ((VariableDeclaration)decl).getSpecifiers());
            // must also get specifiers from the declarator
            Declarator d = ((VariableDeclaration)decl).getDeclarator(0);
            list.addAll(d.getSpecifiers());
            return list;
        } else if (decl instanceof Procedure) {
            return ((Procedure)decl).getReturnType();
        } else {
            return empty_list;
        }
    }

    /**
    * Sets the argument at the specified position with the given new expression.
    *
    * @param n the position of the arugment.
    * @param expr the new argument.
    * @throws IllegalArgumentException if <b>n</b> or <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setArgument(int n, Expression expr) {
        setChild(n + 1, expr);
    }

    /**
    * Sets the argument list with the specified list of new arguments.
    *
    * @param args the list of new arguments.
    * @throws IllegalArgumentException if an element of <b>args</b> is invalid.
    * @throws NotAnOrphanException if an element of <b>args</b> has a parent
    * object.
    */
    public void setArguments(List args) {
        Traversable save = children.get(0);
        children.clear();
        children.add(save);
        for (Object arg : args) {
            addChild((Traversable)arg);
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
    * Sets the name of the function call with the specified new expression.
    *
    * @param expr the expression to be used as a new name.
    * @throws IllegalArgumentException if <b>expr</b> is invalid.
    * @throws NotAnOrphanException if <b>expr</b> has a parent object.
    */
    public void setFunction(Expression expr) {
        setChild(0, expr);
    }

    @Override
    public FunctionCall clone() {
        return (FunctionCall)super.clone();
    }

}
