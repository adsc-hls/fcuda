package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
* Represents a text annotation (comment or pragma) or a list of keyed values.
* Internally, an annotation is a single string <i>or</i> a map of
* keys to values.  There is no restriction on the type or content of the
* keys and values.  Compiler passes are free to use annotations as they see
* fit, although annotations used by multiple passes should be well-documented.
*
* By default, annotations are printed as a multi-line comment. They can also
* be printed as pragmas.  If the text value of the annotation has been set,
* the text is printed, otherwise the list of keyed values is printed.
*
* As of version 1.1 (July, 2009), <b>PreAnnotation</b> is used only internally.
* Pass writers should use {@link Annotatable} interface instead.
*/
public class PreAnnotation extends Declaration {

    private static Method class_print_method;

    /** Useful for passing to setPrintMethod or setClassPrintMethod. */
    public static final Method print_as_comment_method;

    /** Useful for passing to setPrintMethod or setClassPrintMethod. */
    public static final Method print_as_pragma_method;

    /** Useful for passing to setPrintMethod or setClassPrintMethod. */
    public static final Method print_raw_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = PreAnnotation.class;
            params[1] = PrintWriter.class;
            class_print_method =
                    params[0].getMethod("defaultPrint", params);
            print_as_comment_method =
                    params[0].getMethod("printAsComment", params);
            print_as_pragma_method =
                    params[0].getMethod("printAsPragma", params);
            print_raw_method =
                    params[0].getMethod("printRaw", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    protected HashMap map;

    protected String text;

    /**
    * Creates an empty annotation.
    */
    public PreAnnotation() {
        super(-1);
        object_print_method = class_print_method;
        map = new HashMap(4);
        parent = null;
        text = null;
    }

    /**
    * Creates a text annotation.
    *
    * @param text The text to be used as a comment or pragma.
    */
    public PreAnnotation(String text) {
        super(-1);
        object_print_method = class_print_method;
        map = new HashMap(1);
        this.text = text;
    }

    @SuppressWarnings("unchecked")
    public void add(Object key, Object value) {
        map.put(key, value);
    }

    /**
    * Prints an annotation to a stream.
    *
    * @param a The annotation to print.
    * @param o The writer on which to print the annotation.
    */
    public static void defaultPrint(PreAnnotation a, PrintWriter o) {
        printAsComment(a, o);
    }

    @Override
    public boolean equals(Object o) {
        try {
            PreAnnotation a = (PreAnnotation)o;
            return toString().equals(a.toString());
        } catch(ClassCastException e) {
            return false;
        }
    }

    @SuppressWarnings("unchecked")
    public List<IDExpression> getDeclaredIDs() {
        return (List<IDExpression>)empty_list;
    }

    /**
    * Provides access to the annotation map.
    */
    public Map getMap() {
        return map;
    }

    public String getText() {
        return text;
    }

    public int hashCode() {
        return toString().hashCode();
    }

    /**
    * Prints an annotation as a multi-line comment.
    *
    * @param a The annotation to print.
    * @param o The writer on which to print the annotation.
    */
    public static void printAsComment(PreAnnotation a, PrintWriter o) {
        o.println("/*");
        if (a.text != null) {
            o.println(a.text);
        } else {
            o.println(a.map.toString());
        }
        o.print("*/");
    }

    /**
    * Prints an annotation as a single-line pragma.
    *
    * @param a The annotation to print.
    * @param o The writer on which to print the annotation.
    */
    public static void printAsPragma(PreAnnotation a, PrintWriter o) {
        o.print("#pragma ");
        if (a.text != null) {
            o.print(a.text);
        } else {
            o.print(a.map.toString());
        }
    }

    /**
    * Prints an annotation's contents without enclosing them in comments.
    *
    * @param a The annotation to print.
    * @param o The writer on which to print the annotation.
    */
    public static void printRaw(PreAnnotation a, PrintWriter o) {
        if (a.text != null) {
            o.print(a.text);
            o.print(" ");
        } else {
            o.print(a.map.toString());
        }
    }

    /**
    * Unsupported - this object has no children.
    */
    public void setChild(int index, Traversable t) {
        throw new UnsupportedOperationException();
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
    * Sets the text of the annotation.
    *
    * @param text The text to be used as a comment or pragma.
    */
    public void setText(String text) {
        this.text = text;
    }

}
