package cetus.analysis;

import cetus.hir.Expression;
import cetus.hir.Loop;

import java.util.LinkedList;

/**
 * Represents loop-related information
 */
public class LoopInfo {
    private Expression upperBound; // IntLiteral or Variable
    private Expression lowerBound; // IntLiteral or Variable
    private Expression increment;  // IntLiteral or Variable
    private Expression indexVar;
    private LinkedList<Loop> loopNest;
    //set of all enclosing outermost loops and the loop itself
    
    public LoopInfo () {
        this.upperBound = null;
        this.lowerBound = null;
        this.increment = null;
        this.indexVar = null;
        this.loopNest = null;
    }
    
    /**
     * Creates a data structure containing loop-related information (use only
     * if canonical loop)
     * @param loop
     */
    public LoopInfo(Loop loop) {
        this.upperBound = LoopTools.getUpperBoundExpression(loop);
        this.lowerBound = LoopTools.getLowerBoundExpression(loop);
        this.increment = LoopTools.getIncrementExpression(loop);
        this.indexVar = LoopTools.getIndexVariable(loop);
        this.loopNest = LoopTools.calculateLoopNest(loop);
    }
    
    /* Access functions */
    public Expression getLoopUB() {
        return upperBound;
    }
    
    public void setLoopUB(Expression ub) {
        upperBound = ub;
    }
    
    public Expression getLoopLB() {
        return lowerBound;
    }
    
    public void setLoopLB(Expression lb) {
        lowerBound = lb;
    }
    
    public Expression getLoopIncrement() {
        return increment;
    }
    
    public void setLoopIncrement(Expression inc) {
        increment = inc;
    }
    
    public Expression getLoopIndex() {
        return indexVar;
    }
    
    public LinkedList getNest() {
        return loopNest;
    }
    
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(indexVar);
        sb.append(" from ").append(lowerBound);
        sb.append(" to ").append(upperBound);
        sb.append(" step ").append(increment);
        return sb.toString();
    }
}
