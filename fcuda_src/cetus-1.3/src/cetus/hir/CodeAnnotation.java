package cetus.hir;

/**
* CodeAnnotation is used for an annotation of raw code type.
* It is convenient to insert a raw code if pass writers are sure about the
* correctness of the inserted code for their passes. One example is inlined
* library codes that are called after a certain program transformation.
*/
public class CodeAnnotation extends Annotation {

    private static final long serialVersionUID = 3475L;

    /**
    * Constructs a new code annotation with the given code.
    */
    public CodeAnnotation(String code) {
        super();
        put("code", code);
    }

    /**
    * Returns the string representation of this code annotation.
    */
    @Override
    public String toString() {
        if (skip_print) {
            return "";
        } else {
            return get("code");
        }
    }

}
