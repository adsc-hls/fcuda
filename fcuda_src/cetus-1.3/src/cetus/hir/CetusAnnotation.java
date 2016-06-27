package cetus.hir;

import java.util.Map;
import java.util.Set;

/**
* CetusAnnotation is used for internal annotations inserted by Cetus analysis
* and transformation. Parallelization passes usually insert this type of
* annotations.
*/
public class CetusAnnotation extends PragmaAnnotation {

    private static final long serialVersionUID = 3473L;

    /**
    * Constructs an empty cetus annotation.
    */
    public CetusAnnotation() {
        super();
    }

    /**
    * Constructs a cetus annotation with the given key-value pair.
    */
    public CetusAnnotation(String key, Object value) {
        super();
        put(key, value);
    }

    /**
    * Returns a string representation of this cetus annotation.
    * @return a string representation.
    */
    @Override
    public String toString() {
        if (skip_print) {
            return "";
        }
        StringBuilder str = new StringBuilder(80);
        str.append(super.toString()).append("cetus").append(" ");
        if (containsKey("parallel")) {
            str.append("parallel").append(" ");
        }
        for (String key : keySet()) {
            if (key.equals("parallel")) {
                ;
            } else if (key.equals("lastprivate") || key.equals("private")) {
                Set<Symbol> private_set = this.get(key);
                str.append(key).append("(");
                // NOTE: #pragma cetus private may contain different symbols
                // sharing the same name but only one of those are printed. This
                // is an expected behavior so don't panic; it is still possible
                // to differentiate those symbols in a cetus pass.
                str.append(PrintTools.collectionToString(private_set, ", "));
                str.append(") ");
            } else if (key.equals("reduction")) {
                Map<String, Set<Expression>> reduction_map = this.get(key);
                for (String op : reduction_map.keySet()) {
                    str.append("reduction(").append(op).append(": ");
                    str.append(PrintTools.collectionToString(
                            reduction_map.get(op), ", "));
                    str.append(") ");
                }
            } else if (key.equals("use") || key.equals("def")) {
                str.append(key).append(" (");
                str.append(PrintTools.collectionToString(
                        (Set)this.get(key), ", "));
                str.append(") ");
            } else {
		String key_str = get(key);
                str.append(key).append(" ").append(key_str).append(" ");
            }
        }
        return str.toString();
    }

}
