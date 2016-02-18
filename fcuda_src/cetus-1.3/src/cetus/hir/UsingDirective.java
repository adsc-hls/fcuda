package cetus.hir;

import java.util.ArrayList;
import java.util.List;

/** This class is not supported */
public class UsingDirective extends Declaration {

    @SuppressWarnings("unchecked")
    public List<IDExpression> getDeclaredIDs() {
        return (List<IDExpression>) empty_list;
    }
    
    public String toString() {
        return "";
    }

}
