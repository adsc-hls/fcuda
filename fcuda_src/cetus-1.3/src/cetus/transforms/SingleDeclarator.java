package cetus.transforms;

import cetus.hir.*;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.NoSuchElementException;
import java.util.Set;

/**
* Transforms a program such that every declaration contains at most one
* declarator.  The declarations are kept in order, so for example
* {@code "int x, y, z;"} becomes {@code "int x; int y; int z;"}.
*/
public class SingleDeclarator extends TransformPass {

    private static String pass_name = "[SingleDeclarator]";

    public SingleDeclarator(Program program) {
        super(program);
    }

    public String getPassName() {
        return pass_name;
    }

    private void eliminateMultipleDeclarators(VariableDeclaration decl) {
        PrintTools.printlnStatus(3, pass_name, "eliminating multiples in",decl);
        SymbolTable outer = null;
        Traversable child = decl, parent = decl.getParent();
        if (parent instanceof SymbolTable) {
            outer = (SymbolTable)parent;
        } else if (parent instanceof DeclarationStatement) {
            child = parent;
            parent = child.getParent();
            outer = (SymbolTable)parent;
        } else {
            return;
        }
        /* now parent is a symbol table and child is either decl or declstmt. */
        VariableDeclaration placeholder =
                new VariableDeclaration(new LinkedList());
        outer.addDeclarationAfter(decl, placeholder);
        parent.removeChild(child);
        for (int i = decl.getNumDeclarators() - 1; i >= 0; --i) {
            Declarator d = decl.getDeclarator(i);
            outer.addDeclarationAfter(placeholder, new VariableDeclaration(
                    decl.getSpecifiers(), d.clone()));
        }
        if (placeholder.getParent() instanceof DeclarationStatement) {
            parent.removeChild(placeholder.getParent());
        } else {
            parent.removeChild(placeholder);
        }
    }

    public void start() {
        DFIterator<Declaration> iter =
                new DFIterator<Declaration>(program, Declaration.class);
        while (iter.hasNext()) {
            Declaration d = iter.next();
            if (d instanceof Procedure) {
                PrintTools.printlnStatus(1, pass_name, "examining procedure",
                        "\"", ((Procedure)d).getName(), "\"");
            } else if (d instanceof VariableDeclaration) {
                if (d.getChildren().size() > 1) {
                    eliminateMultipleDeclarators((VariableDeclaration)d);
                }
            }
        }
    }
}
