package cetus.hir;

import java.io.PrintWriter;

/**
* Represents the name of a user-defined type, such as created by a typedef,
* class declaration, etc.
*/
public class UserSpecifier extends Specifier {

    private IDExpression usertype;

    public UserSpecifier(IDExpression usertype) {
        this.usertype = usertype;
    }
    
    public IDExpression getIDExpression() {
        return usertype;
    }

    public boolean isStructure() {
        return toString().startsWith("struct ");
    }

    public void print(PrintWriter o) {
        usertype.print(o);
    }

    public boolean equals(Object o) {
        return (o != null &&
                o instanceof UserSpecifier &&
                ((UserSpecifier)o).usertype.equals(usertype));
    }

}
