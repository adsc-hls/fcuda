package cetus.application;

import cetus.analysis.DFANode;
import cetus.hir.Expression;
import cetus.hir.Traversable;
import java.util.List;
import java.util.Set;

/**
 * A class that implements UseDefChain provides the Use-Def chain information
 * which is computed from the control flow graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public interface UseDefChain {
    public List<Traversable> getDefList(Expression use);
    public Set<DFANode> getDefDFANodeSet(Expression use);
    public List<Traversable> getLocalDefList(Expression use);
}
