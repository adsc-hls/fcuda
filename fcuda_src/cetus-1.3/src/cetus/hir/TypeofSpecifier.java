package cetus.hir;

import java.io.PrintWriter;
import java.util.List;

/** This class is no longer supported. */
public class TypeofSpecifier extends Specifier {

    private Expression expr;

    private List type_id;

    public TypeofSpecifier(Expression expr) {
        this.expr = expr;
        this.type_id = null;
    }
    
    public TypeofSpecifier(List type_id) {
        this.expr = null;
        this.type_id = type_id;
    }

    public void print(PrintWriter o) {
        o.print("typeof ( ");
        if (expr != null) {
            expr.print(o);
        } else {
            PrintTools.printListWithSpace(type_id, o);
        }
        o.print(" ) ");
    }

}
