package fcuda.transforms;

import java.util.*;

import fcuda.utils.*;
import fcuda.common.*;

import cetus.hir.*;
import cetus.analysis.*;

/* This class converts loops that iterate based on thread-dependent conditions into eternal
 * while loops with thread-dependent if clauses.
 *
 * This transformation is mainly done to enable thread-loop unrolling in a later pass
 * The main reason is that AutoPilot does not parallelize sequential loops even though
 * they might not have any dependencies, but it can parallelize if statements.
 *
 */

public class RemoveThrDepLoops extends KernelTransformPass
{
  private Procedure mProcedure;


  public String getPassName()
  {
    return new String("[RemoveThrDepLoops-FCUDA]");
  }

  public RemoveThrDepLoops(Program program)
  {
    super(program);
    mProcedure = null;
  }

  private boolean isThrDepLoop(ForLoop flp)
  {
    boolean thrDep = false;
    Statement iniStmt = flp.getInitialStatement();
    Expression cond = flp.getCondition();
    Expression update = flp.getStep();

    Set<Expression> useSet = DataFlow.getUseSet(iniStmt);
    useSet.addAll(DataFlow.getUseSet(cond));
    useSet.addAll(DataFlow.getUseSet(update));

    if (useSet.contains(MCUDAUtils.Tidx.getId().get(0)))
      thrDep = true;

    if (!thrDep) {
      for (Expression expr : useSet) {
        if (!(expr instanceof IDExpression))
          continue;
        if ((FCUDAutils.getThrDefExp((IDExpression)expr) != null) ||
            (FCUDAutils.getThrDepExp((IDExpression)expr) != null)) {
          thrDep = true;
          break;
            }
      }
    }
    return thrDep;
  }


  private void convert(ForLoop flp)
  {
    Statement initStmt = flp.getInitialStatement();
    Expression cond = flp.getCondition();
    Expression update = flp.getStep();
    CompoundStatement parentBody = (CompoundStatement) Tools.getAncestorOfType(flp, CompoundStatement.class);

    System.out.println("Converting ForLoop: ("+initStmt.toString()+" ; "+cond.toString()+" ; "+update.toString()+")");

    // Handle Initial statement
    if (initStmt instanceof ExpressionStatement &&
        ((ExpressionStatement)initStmt).getExpression() instanceof CommaExpression) {
      CommaExpression ce = (CommaExpression)((ExpressionStatement)initStmt).getExpression();
      Statement curStmt  = null;

      // Create Loop Initialization statements
      for (Traversable t : ce.getChildren() ) {
        curStmt = new ExpressionStatement((Expression)t);
        parentBody.addStatementBefore(flp, curStmt);
      }
    } else {
      parentBody.addStatementBefore(flp, initStmt.clone());
    }


    // Handle While Loop Body
    CompoundStatement whileBody = new CompoundStatement();
    IfStatement exitIFstmt = new IfStatement(Symbolic.negate((Expression)cond.clone()),
        new BreakStatement());
    whileBody.addStatement(exitIFstmt);
    FCUDAGlobalData.addThrCondExit(exitIFstmt);

    CompoundStatement tmpCstmt = new CompoundStatement();
    IfStatement mainIFstmt = new IfStatement((Expression)cond.clone(), tmpCstmt);
    tmpCstmt.swapWith(flp.getBody());

    whileBody.addStatement(mainIFstmt);

    // Handle Update statement
    CompoundStatement mainIFbody = (CompoundStatement)mainIFstmt.getThenStatement();
    if (update instanceof CommaExpression) {
      for (Traversable t : update.getChildren()) {
        ExpressionStatement updateStmt = new ExpressionStatement((Expression)t);
        mainIFbody.addStatement(updateStmt);
      }
    } else
      mainIFbody.addStatement(new ExpressionStatement(update.clone()));

    // Handle While Loop statement
    WhileLoop whileStmt = new WhileLoop(new IntegerLiteral(1), whileBody);
    parentBody.addStatementBefore(flp, whileStmt);
    flp.detach();

  }

  public void transformProcedure(Procedure proc)
  {
    mProcedure = proc;
    FCUDAGlobalData.recordDataDeps(proc);

    // Iterate through the procedure statements
    PostOrderIterator stIter = new PostOrderIterator(proc);
    stIter.pruneOn(Expression.class);

    while (stIter.hasNext()) {

      Statement stmt = null;

      try {
        stmt = (Statement)stIter.next(Statement.class);
      } catch (NoSuchElementException e) {
        break;
      }

      // Only interested in for-loops
      if (!(stmt instanceof ForLoop))
        continue;
      if (isThrDepLoop((ForLoop)stmt))
        convert((ForLoop)stmt);
    }
  }
}
