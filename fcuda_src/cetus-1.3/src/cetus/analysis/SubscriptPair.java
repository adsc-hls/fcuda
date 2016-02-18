package cetus.analysis;

import cetus.hir.*;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * Creates a pair of affine subscripts where subscript is a single dimension of
 * an array reference
 */
public class SubscriptPair {
    /* Store normalized expression if affine */
    private Expression subscript1, subscript2;
    /* Statements that contain the subscripts */
    private Statement stmt1, stmt2;
    /* Loops from which indices are present in the subscript pair */
    private LinkedList<Loop> present_loops;
    /* All loops from the enclosing loop nest */
    private LinkedList<Loop> enclosing_loops;
    /* Loop information for the enclosing loop nest */
    private HashMap<Loop, LoopInfo> enclosing_loops_info;

    /**
    * Constructs a new subscript pair with the given pairs of expressions,
    * statements, and the loop nest information.
    * @param s1 the first subscript expression.
    * @param s2 the second subscript expression.
    * @param st1 the first statement.
    * @param st2 the second subscript expression.
    * @param nest the loop nest.
    * @param loopinfo the extra information about loops.
    */
    public SubscriptPair(
            Expression s1, Expression s2, // Two subscripts (possibly orphans)
            Statement st1, Statement st2, // Two statements containing s1,s2
            LinkedList<Loop> nest, HashMap <Loop, LoopInfo> loopinfo) {
        // All symbols present in affine expressions
        List<Identifier> symbols_in_expressions;
        List symbols_in_s1, symbols_in_s2;
        this.subscript1 = s1;
        this.subscript2 = s2;
        this.stmt1 = st1;
        this.stmt2 = st2;
        this.enclosing_loops = nest;
        this.enclosing_loops_info = loopinfo;
        Set<Symbol> symbols_s1 = DataFlowTools.getUseSymbol(s1);
        Set<Symbol> symbols_s2 = DataFlowTools.getUseSymbol(s2);
        present_loops = new LinkedList<Loop>();        
        for (Loop loop: nest) {
            LoopInfo info = loopinfo.get(loop);
            Expression index = info.getLoopIndex();
            if (symbols_s1.contains(((Identifier)index).getSymbol()) ||
                    symbols_s2.contains(((Identifier)index).getSymbol())) {
                present_loops.addLast(loop);
            }
        }
    }

    protected HashMap<Loop, LoopInfo> getEnclosingLoopsInfo() {
        return enclosing_loops_info;
    }
    
    protected LinkedList<Loop> getEnclosingLoopsList() {
        return enclosing_loops;
    }
    
    protected LinkedList<Loop> getPresentLoops() {
        return present_loops;
    }
    
    protected Expression getSubscript1() {
        return subscript1;
    }
    
    protected Expression getSubscript2() {
        return subscript2;
    }

    protected Statement getStatement1() {
        return stmt1;
    }

    protected Statement getStatement2() {
        return stmt2;
    }
    
    protected int getComplexity() {
        return present_loops.size();
    }

    public String toString() {
        StringBuilder str = new StringBuilder(80);
        str.append("[SUBSCRIPT-PAIR] ").append(subscript1);
        str.append(", ").append(subscript2).append(PrintTools.line_sep);
        for (Loop loop : enclosing_loops) {
            str.append("  enclosed by ").append(enclosing_loops_info.get(loop));
            str.append(PrintTools.line_sep);
        }
        for (Loop loop : present_loops) {
            str.append("  relevant with ");
            str.append(enclosing_loops_info.get(loop));
            str.append(PrintTools.line_sep);
        }
        return str.toString();
    }
}
