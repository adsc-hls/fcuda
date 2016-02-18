package cetus.application;

import cetus.hir.Expression;
import cetus.hir.Traversable;
import java.util.List;

/**
 * A class that implements DefUseChain provides the Def-Use chain information
 * which is computed from the control flow graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public interface DefUseChain {
    public List<Traversable> getUseList(Expression def);
    public List<Traversable> getLocalUseList(Expression def);
    public boolean isReachable(Expression def, Expression use);
}
