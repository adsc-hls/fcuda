package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
* AnnotationDeclaration is used for stand-alone annotations in non-executable
* code section, e.g., in a TranslationUnit.
*/
public class AnnotationDeclaration extends Declaration {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = AnnotationDeclaration.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException ex) {
            throw new InternalError();
        }
    }

    /**
    * Constructs an empty annotation declaration.
    */
    public AnnotationDeclaration() {
        super(-1);
        object_print_method = class_print_method;
    }

    /**
    * Constructs a new annotation declaration with the given annotation.
    *
    * @param annotation the new annotation to be inserted.
    */
    public AnnotationDeclaration(Annotation annotation) {
        super(-1);
        object_print_method = class_print_method;
        annotate(annotation);
    }

    /**
    * Returns an empty list - not used for an annotation declaration.
    */
    @SuppressWarnings("unchecked")
    public List<IDExpression> getDeclaredIDs() {
        return (List<IDExpression>)empty_list;
    }

    /**
    * Prints this annotation declaration. It does not print anything since an
    * annotation declaration works like a place holder for the enclosed
    * annotations.
    */
    public static void defaultPrint(AnnotationDeclaration d, PrintWriter o) {
        // nothing to print.
    }

    /**
    * Sets the class print method with the specified method.
    */
    public static void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

}
