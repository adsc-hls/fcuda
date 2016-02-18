package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.List;

/**
* Represents a container that holds a raw code not expressed as a well-defined
* internal representation. Non-standard GCC extension such as inline assembly
* is represented with this expression.
*/
public class SomeExpression extends Expression {

    private static Method class_print_method;

    static {
        try {
            class_print_method = SomeExpression.class.getMethod("defaultPrint",
                    new Class<?>[] {SomeExpression.class, PrintWriter.class});
        } catch(NoSuchMethodException e) {
            throw new InternalError("No printing method defined");
        }
    }

    /** The raw code stored in the expression */
    private String code;

    private SomeExpression() {
        code = "";
        object_print_method = class_print_method;
    }

    /**
    * Constructs some unknown expression having the specified raw code and the
    * given list of children.
    * @param code the raw code for the expression.
    * @param children the child expressions.
    */
    public SomeExpression(String code, List<Traversable> children) {
        object_print_method = class_print_method;
        this.code = code;
        for (Traversable child : children) {
            this.addChild(child);
        }
    }

    @Override
    public SomeExpression clone() {
        SomeExpression ret = (SomeExpression)super.clone();
        ret.code = this.code;
        return ret;
    }

    /** The default print method just prints out the raw code */
    public static void defaultPrint(SomeExpression e, PrintWriter o) {
        o.print(e.code);
    }

    @Override
    public boolean equals(Object o) {
        return (o != null && this.toString().equals(o.toString()));
    }

}
