package cetus.transforms;

import cetus.analysis.IPPointsToAnalysis;
import cetus.analysis.IPRangeAnalysis;
import cetus.analysis.ArrayParameterAnalysis;
import cetus.analysis.RangeAnalysis;
import cetus.hir.*;

/**
* Base class of all transformation passes. For consistent compilation, there
* are a series of checking processes at the end of every transformation pass.
*/
public abstract class TransformPass {

    /** The associated program */
    protected Program program;

    /** Flags for skipping consistency checking */
    protected boolean disable_protection;

    /** Flags for skipping analysis invalidation */
    protected boolean disable_invalidation;

    /** Constructs a transform pass with the given program */
    protected TransformPass(Program program) {
        this.program = program;
        disable_protection = false;
        disable_invalidation = false;
    }

    /** Returns the name of the transform pass */
    public abstract String getPassName();

    /** 
    * Invokes the specified transform pass.
    * @param pass the transform pass that is to be run.
    */
    public static void run(TransformPass pass) {
        double timer = Tools.getTime();
        PrintTools.println(pass.getPassName() + " begin", 0);
        pass.start();
        PrintTools.println(pass.getPassName() + " end in " +
                String.format("%.2f seconds", Tools.getTime(timer)), 0);
        if (!pass.disable_protection) {
            if (!IRTools.checkConsistency(pass.program)) {
                throw new InternalError("Inconsistent IR after " +
                                        pass.getPassName());
            }
            SymbolTools.linkSymbol(pass.program);
        }
        // Invalidates points-to relations.
        // TODO: what about ddgraph ?
        if (!pass.disable_invalidation) {
            IPPointsToAnalysis.clearPointsToRelations();
            IPRangeAnalysis.clear();
            ArrayParameterAnalysis.invalidate();
            RangeAnalysis.invalidate();
        }
    }

    /** Starts a transform pass */
    public abstract void start();

}
