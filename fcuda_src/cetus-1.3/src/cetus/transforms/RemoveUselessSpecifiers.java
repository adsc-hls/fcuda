package cetus.transforms;

import cetus.hir.*;

import java.util.HashSet;
import java.util.NoSuchElementException;
import java.util.Set;

public class RemoveUselessSpecifiers {

    private static String pass_name = "[RemoveUselessSpecifiers]";

    private Program program;

    private RemoveUselessSpecifiers(Program program) {
        this.program = program;
    }
    
    public static void run(Program program) {
        PrintTools.printlnStatus(pass_name + " begin", 1);
        RemoveUselessSpecifiers pass = new RemoveUselessSpecifiers(program);
        pass.start();
        PrintTools.printlnStatus(pass_name + " end", 1);
    }

    @SuppressWarnings("unchecked")
    private void start() {
        DepthFirstIterator i = new DepthFirstIterator(program);
        i.pruneOn(VariableDeclaration.class);
        Set<Class> set = new HashSet<Class>();
        set.add(Procedure.class);
        set.add(VariableDeclaration.class);
        for (;;) {
            Procedure proc = null;
            VariableDeclaration decl = null;
            try {
                Object o = i.next(set);
                if (o instanceof Procedure) {
                    proc = (Procedure)o;
                } else {
                    decl = (VariableDeclaration)o;
                }
            } catch(NoSuchElementException e) {
                break;
            }
            if (proc != null) {
                PrintTools.printlnStatus(pass_name +
                                         " examining procedure " +
                                         proc.getName(), 2);
            } else {
                while (decl.getSpecifiers().remove(Specifier.AUTO));
                while (decl.getSpecifiers().remove(Specifier.REGISTER));
            }
        }
    }

}
