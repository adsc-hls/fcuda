package cetus.hir;

import java.io.PrintWriter;

/**
* Any class implementing this interface can print its data. Every class provides
* a default way of printing itself as source code.  The class may also support
* overriding the default behavior at the class level.  For example, the pass
* writer may wish all ForStatements to be printed in a certain way.  It may also
* support overriding the behavior for a particular object.  For example, once a
* pass determines that a loop is parallel, it may need to modify the print
* behavior for that loop only.  For consistency, changing the print method of a
* class or object will also affect the behavior of toString for that class or
* object.
*/
public interface Printable {

    /**
    * Print the code for the IR represented by the object.  Always calls
    * object_print_method(this, o).  If the object's print method is null,
    * nothing is printed; this provides an easy mechanism to temporarily hide
    * something.
    *
    * @param o The writer on which to print the data.
    */
    void print(PrintWriter o);

}
