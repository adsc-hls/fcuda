package cetus.hir;

/**
* CommentAnnotation is used for an annotation of comment type. It provides two
* different print method depending on the location of the comments.
*/
public class CommentAnnotation extends Annotation {

    private static final long serialVersionUID = 3476L;

    private boolean one_liner;  // Is this printed in one line.

    /**
    * Constructs a new comment annotation with the given comment.
    */
    public CommentAnnotation(String comment) {
        super();
        // Normalize the comment to avoid weird reconstruction of C++ comments.
        // e.g. '//* comment *//' becomes '/* comment */' rather than
        // '/* *comment *// */'
        comment = comment.replaceFirst(" *[\\/\\*]+", "");
        comment = comment.replaceFirst(" *[\\/\\*]+$", "");
        // Additional normalization to avoid termination mark '*/'.
        comment = comment.replaceAll("\\*/", "");
        put("comment", comment);
        one_liner = false;
    }

    /**
    * Sets the one_liner field; true for one-liner comments, false for
    * normal multi-line comments.
    */
    public void setOneLiner(boolean one_liner) {
        this.one_liner = one_liner;
    }

    /**
    * Returns the string representation of this comment.
    * @return the string comments.
    */
    @Override
    public String toString() {
        if (skip_print) {
            return "";
        } else if (one_liner) {
            return ("/* " + get("comment") + " */");
        } else {
            return ("/*\n" + get("comment") + "\n*/");
        }
    }

}
