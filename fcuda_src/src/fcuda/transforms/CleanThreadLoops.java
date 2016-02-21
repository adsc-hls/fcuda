//============================================================================//
//    FCUDA
//    Copyright (c) <2016> 
//    <University of Illinois at Urbana-Champaign>
//    <University of California at Los Angeles> 
//    All rights reserved.
// 
//    Developed by:
// 
//        <ES CAD Group & IMPACT Research Group>
//            <University of Illinois at Urbana-Champaign>
//            <http://dchen.ece.illinois.edu/>
//            <http://impact.crhc.illinois.edu/>
// 
//        <VAST Laboratory>
//            <University of California at Los Angeles>
//            <http://vast.cs.ucla.edu/>
// 
//        <Hardware Research Group>
//            <Advanced Digital Sciences Center>
//            <http://adsc.illinois.edu/>
//============================================================================//

package fcuda.transforms;

import java.util.*;

import fcuda.ir.*;
import fcuda.common.*;
import fcuda.utils.*;
import fcuda.analysis.*;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.*;

/**
 * Filter thread-independent statements within thread-loops and  
 * move them outside of thread-lopps.  
 * If affected thread-loop becomed empty, remove it from task 
 * function.
 */
public class CleanThreadLoops extends KernelTransformPass
{
  private Procedure mProcedure;

  private HashMap<ThreadLoop, List<Statement>> mTloopIndepStmts;

  public String getPassName()
  {
    return new String("[CleanThreadLoops-MCUDA]");
  }

  public CleanThreadLoops(Program program)
  {
    super(program);
    mTloopIndepStmts = new HashMap();
    mTloopIndepStmts.clear();
  }



  private boolean isPreUsed(Statement stmt, CompoundStatement body, 
      List<Statement> indStmts)
  {
    boolean preUse = false;
    Set<Expression> defs = DataFlow.mayDefine(stmt); 

    for (Expression dExp : defs) {
      if (!(dExp instanceof IDExpression))
        continue;

      IDExpression def = (IDExpression) dExp;

      int idx = Tools.indexByReference(body.getChildren(), stmt);
      if (idx == -1)
        throw new IllegalStateException();

      int id;
      for (id = 0; id<idx; ++id) {
        Statement prevStmt = (Statement) body.getChildren().get(id);
        if (indStmts.contains(prevStmt))  // Assume all indep statements will be shifted to the same direction
          continue;
        if (DataFlow.getUseSet(prevStmt).contains(def)) {
          preUse = true;
          break;
        }
      }
      if (preUse == true)
        break;

    }

    return preUse;

  }

  private List<Statement> getThrIndepStmts(ThreadLoop tloop, Procedure proc) 
  {
    boolean isDep;
    LinkedList<Statement> indepStmts = new LinkedList();
    indepStmts.clear();

    CompoundStatement tlpBody = (CompoundStatement) tloop.getBody();

    for (Traversable child : tlpBody.getChildren()) {
      isDep = false;

      // Currently only handle scalar assignments
      if(!(child instanceof ExpressionStatement))
        continue;

      Expression expr = ((ExpressionStatement)child).getExpression();

      // Only look into assignments
      if (!((expr instanceof AssignmentExpression) || (expr instanceof UnaryExpression)))
        continue;

      Set<Expression> defs = DataFlow.mayDefine(expr);
      for (Expression dExp : defs) {
        if (dExp instanceof ArrayAccess) {
          isDep = true;
          break;
        }

        if (!(dExp instanceof IDExpression))
          continue;

    
        FCUDAGlobalData.recordDataDeps(proc);
        Set<Expression> defExprs = FCUDAGlobalData.getVarDependences(proc, (IDExpression)dExp);
        if (defExprs == null) {
          isDep = true;
          break;
        }

        if (defExprs.contains(MCUDAUtils.Tidx.getId().get(0))) {
          isDep = true;
          break;
        }
      }
      if (isDep == false)
        indepStmts.add((Statement) child);			

    }
    return indepStmts;
  }


  private void cleanThrLoops(Procedure proc)
  {
    DepthFirstIterator iter = new DepthFirstIterator(proc);
    while(iter.hasNext()) {
      ThreadLoop tloop;

      try {
        tloop = (ThreadLoop)iter.next(ThreadLoop.class);
      } catch (NoSuchElementException e) {
        break;
      }
      List<Statement> thrIndepStmts;

      thrIndepStmts = getThrIndepStmts(tloop, proc);
      mTloopIndepStmts.put(tloop, thrIndepStmts);

    }

    // *AP* Only handles independent statements in the beginning or the end 
    // of the thread loop. Does not handle when in the middle (i.e. uses of
    // defined vars both before and after independent statement)
    for (ThreadLoop tlp : mTloopIndepStmts.keySet()) {
      CompoundStatement parntBody = (CompoundStatement)Tools.getAncestorOfType(tlp, CompoundStatement.class);
      for (Statement stmt : mTloopIndepStmts.get(tlp)) {
        boolean preUsed = isPreUsed(stmt, (CompoundStatement)tlp.getBody(), mTloopIndepStmts.get(tlp));
        stmt.detach();

        if (preUsed)
          parntBody.addStatementAfter(tlp, stmt);
        else
          parntBody.addStatementBefore(tlp, stmt);
      }
      if (tlp.getBody().getChildren().isEmpty())
        ((Statement)tlp).detach();
    }
    // *TAN* clean the list before exit so that 
    //   it will not re-run the statements of other procs	
    mTloopIndepStmts.clear();

  }


  public void transformProcedure(Procedure proc)
  {
    mProcedure = proc;
    List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc.getSymbolName()); 
    for( Procedure task : tskLst ) 
      if(FCUDAutils.getTaskType(task).equals("compute")) {
        cleanThrLoops( task);
      }
  }
}
