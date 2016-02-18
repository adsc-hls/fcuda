package cetus.transforms;

import cetus.exec.Driver;
import cetus.hir.DFIterator;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import java.util.HashSet;

public abstract class ProcedureTransformPass extends TransformPass {

    protected ProcedureTransformPass(Program program) {
        super(program);
    }

    public abstract void transformProcedure(Procedure proc);

    public void start() {
        HashSet skip_set = Driver.getSkipProcedureSet();
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        while (iter.hasNext()) {
              Procedure proc = iter.next();
              if (!skip_set.contains(proc.getName().toString())) {
                  PrintTools.printlnStatus(1,
                        getPassName(), "examining procedure", proc.getName());
                    transformProcedure(proc);
              } else {
                  PrintTools.printlnStatus(1,
                        getPassName(), "skipping procedure", proc.getName());
              }
        }
    }
}
