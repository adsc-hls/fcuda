package cetus.analysis;

import cetus.hir.DFIterator;
import cetus.hir.Loop;
import cetus.hir.Program;

public abstract class LoopAnalysisPass extends AnalysisPass {

    protected LoopAnalysisPass(Program program) {
        super(program);
    }

    public abstract void analyzeLoop(Loop loop);

    public void start() {
        DFIterator<Loop> iter = new DFIterator<Loop>(program, Loop.class);
        while (iter.hasNext()) {
            analyzeLoop(iter.next());
        }
    }
}
