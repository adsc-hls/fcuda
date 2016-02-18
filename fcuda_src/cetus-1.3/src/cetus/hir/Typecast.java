package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/** Represents a typecast expression in C programs. */
public class Typecast extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Typecast.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** Represents a cast type */
    public static class Cast {

        private static String[] name = {"", "dynamic_cast", "static_cast",
                "reinterpret_cast", "const_cast"};

        private int value;

        /** Constructs a cast type with the specified value */
        public Cast(int value) {
            this.value = value;
        }

        /** Prints the cast type */
        public void print(PrintWriter o) {
            o.print(name[value]);
        }

    }

    public static final Cast NORMAL = new Cast(0);
    public static final Cast DYNAMIC = new Cast(1);
    public static final Cast STATIC = new Cast(2);
    public static final Cast REINTERPRET = new Cast(3);
    public static final Cast CONST = new Cast(4);

    private Cast kind;

    private List specs;

    /**
    * Create a normal typecast.
    *
    * @param specs A list of type specifiers.
    * @param expr The expression to cast.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    @SuppressWarnings("unchecked")
    public Typecast(List specs, Expression expr) {
        object_print_method = class_print_method;
        kind = NORMAL;
        this.specs = new ArrayList(specs);
        addChild(expr);
    }

    /**
    * Create a special typecast.
    *
    * @param kind One of <var>NORMAL, DYNAMIC, STATIC, REINTERPRET,</var> or
    *   <var>CONST</var>.
    * @param specs A list of type specifiers.
    * @param expr The expression to cast.
    */
    @SuppressWarnings("unchecked")
    public Typecast(Cast kind, List specs, Expression expr) {
        object_print_method = class_print_method;
        this.kind = kind;
        this.specs = new ArrayList(specs);
        addChild(expr);
    }

    /**
    * Constructs a typecast with the specified kind, specifier, and list of
    * expressions.
    */
    @SuppressWarnings("unchecked")
    public Typecast(Cast kind, Specifier spec, List expr_list) {
        super(expr_list.size());
        object_print_method = class_print_method;
        this.kind = kind;
        this.specs = new ArrayList(1);
        this.specs.add(spec);
        for (int i = 0; i < expr_list.size(); i++) {
            addChild((Traversable) expr_list.get(i));
        }
    }

    /**
    * Prints a typecast expression to a stream.
    *
    * @param c The cast to print.
    * @param o The writer on which to print the cast.
    */
    public static void defaultPrint(Typecast c, PrintWriter o) {
        if (c.needs_parens) {
            o.print("(");
        }
        if (c.kind == NORMAL) {
            if (c.children.size() == 1) {
                o.print("(");
                PrintTools.printListWithSpace(c.specs, o);
                o.print(")");
                c.children.get(0).print(o);
            } else {
                PrintTools.printListWithSpace(c.specs, o);
                o.print("(");
                PrintTools.printListWithSeparator(c.children, o, ",");
                o.print(")");
            }
        } else {
            c.kind.print(o);
            o.print("<");
            PrintTools.printList(c.specs, o);
            o.print(">(");
            c.children.get(0).print(o);
            o.print(")");
        }
        if (c.needs_parens) {
            o.print(")");
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        if (needs_parens) {
            sb.append("(");
        }
        if (kind == NORMAL) {
            if (children.size() == 1) {
                sb.append("(");
                sb.append(PrintTools.listToString(specs, " "));
                sb.append(")");
                sb.append(children.get(0));
            } else {
                sb.append(PrintTools.listToString(specs, " "));
                sb.append("(");
                sb.append(PrintTools.listToString(children, ","));
                sb.append(")");
            }
        } else {
            sb.append(kind);
            sb.append("<");
            sb.append(PrintTools.listToString(specs, ""));
            sb.append(">(");
            sb.append(children.get(0));
            sb.append(")");
        }
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
        if (kind == NORMAL) {
            if (children.size() == 1) {
                h = 31 * h + '(';
                h = hashCode(specs, " ", h);
                h = 31 * h + ')';
                h = ((Expression) children.get(0)).hashCode(h);
            } else {
                h = hashCode(specs, " ", h);
                h = 31 * h + '(';
                h = hashCode(children, ",", h);
                h = 31 * h + ')';
            }
        } else {
            h = hashCode(kind, h);
            h = 31 * h + '<';
            h = hashCode(specs, "", h);
            h = 31 * h + '>';
            h = 31 * h + '(';
            h = ((Expression) children.get(0)).hashCode(h);
            h = 31 * h + ')';
        }
        if (needs_parens) {
            h = 31 * h + ')';
        }
        return h;
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

    /** Returns the list of specifiers of the typecast. */
    public List getSpecifiers() {
        return specs;
    }

    /** Returns the child expression of the typecast. */
    public Expression getExpression() {
        return (Expression)children.get(0);
    }

    /** Compares the typecast with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && kind == ((Typecast)o).kind &&
                // specs may contain an abstract variable declarator, so direct
                // comparison of the list may cause a problem.
                specs.toString().equals(((Typecast)o).specs.toString()));
    }

}
