package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
* AnnotationStatement is used for stand-alone annotations in executable
* code section, e.g., in a CompoundStatement.
*/
public class AnnotationStatement extends Statement {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = AnnotationStatement.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException ex) {
            throw new InternalError();
        }
    }

    /**
    * Constructs an empty annotation statement.
    */
    public AnnotationStatement() {
        super(-1);
        object_print_method = class_print_method;
    }

    /**
    * Constructs a new annotation statement with the specified annotation.
    * @param annotation the new annotation to be inserted.
    */
    public AnnotationStatement(Annotation annotation) {
        super(-1);
        object_print_method = class_print_method;
        annotate(annotation);
    }

    /**
    * Prints this annotation statement. There is nothing to print since the
    * annotation statement is just a place holder.
    * @param s the annotation statement to be printed.
    * @param o the target print writer.
    */
    public static void defaultPrint(AnnotationStatement s, PrintWriter o) {
        // nothing to print.
    }

    /**
    * Sets the class print method with the given method.
    */
    public static void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /** Returns a clone of the annotation statement */
    @Override
    public AnnotationStatement clone() {
        AnnotationStatement as = (AnnotationStatement)super.clone();
        return as;
    }

}
