package fcuda.transforms;

import java.util.*;
import fcuda.common.*;
import fcuda.utils.*;
import fcuda.ir.*;

import cetus.hir.*;
import cetus.exec.*;

/**
 * Enforces synchronization commands in a sequentialized version of the
 * threaded code
 * Assumes that the CUDA programming guide is followed:
 * synchronization points affected by control flow assume that
 * the control flow is thread independent within a block
 * You need to run the SerializeThreads pass before this makes sense.
 */

public class EnforceSyncs extends KernelTransformPass
{
  public String getPassName()
  {
    return new String("[EnforceSyncs-MCUDA]");
  }

  public EnforceSyncs(Program program)
  {
    super(program);
  }

  private Procedure mProcedure;

  public List<Statement> afterSync(Statement sync)
  {
    LinkedList<Statement> statements = new LinkedList<Statement>();
    CompoundStatement scope = (CompoundStatement)sync.getParent();
    boolean foundSync = false;

    for (Traversable t : scope.getChildren()) {
      Statement s = (Statement) t;
      if (s == sync)
        foundSync = true;
      //Children of CS are all statements

      if (!(s instanceof DeclarationStatement) && foundSync)
        statements.add(s);
    }
    return statements;
  }

  public List<Statement> beforeSync(Statement sync)
  {
    LinkedList<Statement> statements = new LinkedList<Statement>();
    CompoundStatement scope = (CompoundStatement)sync.getParent();

    for (Traversable t : scope.getChildren()) {
      //Children of CS are all statements
      Statement s = (Statement) t;
      if (s == sync)
        break;
      if (!(s instanceof DeclarationStatement))
        statements.add(s);
    }
    return statements;
  }

  private boolean statementIsSync(Statement stmt)
  {
    return (stmt instanceof ExpressionStatement) &&
      (((ExpressionStatement)stmt).getExpression().compareTo(MCUDAUtils.getSync()) == 0);
  }

  public void syncOnStmt(Statement sync) throws UnsupportedOperationException
  {
    int numDims = FCUDAGlobalData.getKernTblkDim(mProcedure);
    //We can be sure that the parent is a statement
    CompoundStatement enclosure = MCUDAUtils.getThreadLoopBody(sync);
    if (enclosure == null)
      return;

    if(enclosure.getChildren() == null)
      throw new InternalError();

    ThreadLoop outerLoopOne = MCUDAUtils.getOuterThreadLoop(enclosure);
    if (enclosure.getChildren().contains(sync)) {
      //Direct child of an existing thread loop.
      //Strip all subsequent statements into a new loop.
      CompoundStatement bodyTwo = new CompoundStatement();
      //ThreadLoop outerLoopOne = MCUDAUtils.getOuterThreadLoop(enclosure);
      CompoundStatement threadLoopContext = (CompoundStatement)outerLoopOne.getParent();

      List<DeclarationStatement> bumped_decls = MCUDAUtils.getScopeDecls(enclosure);
      List<Statement> split_stmts = afterSync(sync);
      //TODO: we really should just create another compound statement
      // to correctly represent the scope of these variables
      for (DeclarationStatement s : bumped_decls) {
        s.detach();
        threadLoopContext.addANSIDeclaration(s.getDeclaration().clone());
      }

      //Move regular statements after the sync to the next thread loop
      for (Statement s : split_stmts) {
        s.detach();
        bodyTwo.addStatement(s);
      }

      sync.detach();

      threadLoopContext.addStatementAfter(outerLoopOne, sync);

      if (enclosure.countStatements() == 0) {
        //Nothing to compute in this loop.  Get rid
        // of it.
        outerLoopOne.detach();
      }

      if (bodyTwo.countStatements() != 0) {
        ThreadLoop outerLoopTwo = MCUDAUtils.NewNestedThreadLoop(numDims, bodyTwo);

        // **AP** Copy update/bound expressions from original threadLoop (Required in FCUDA)
        for (int dim = 0; dim < numDims; dim++) {
          outerLoopTwo.setUpdate(dim, (Expression) outerLoopOne.getUpdate(dim).clone());
          outerLoopTwo.setBound(dim, (Expression) outerLoopOne.getBound(dim).clone());
        }

        threadLoopContext.addStatementAfter(sync, outerLoopTwo);
      }

      //Every thread loop needs a synchronization immediately after it
      // If the statement was not a CUDA sync and the first thread loop exists, add a CUDA sync
      if (!statementIsSync(sync)) {
        if (enclosure.countStatements() != 0)
          threadLoopContext.addStatementAfter(outerLoopOne,
              new ExpressionStatement(MCUDAUtils.getSync()));
      } else {
        // If the statement is a CUDA sync and the first thread loop does not exist, remove it
        if (enclosure.countStatements() == 0)
          sync.detach();
      }
      //The second thread loop inherits the sync from the preexisting loop just split

    } else {
      //There is other control stuff going on between the sync
      // statement and the thread loop.  So, recursively,
      // break this context, and then break parent contexts until
      // we get to the enclosing thread loop.
      CompoundStatement immediateParent = (CompoundStatement)sync.getParent();
      Statement controlStruct = (Statement)immediateParent.getParent();
      if (controlStruct instanceof Loop)
        (new UniformLoops(program)).transformLoop((Loop)controlStruct);
      
      CompoundStatement threadLoopBodyOne = new CompoundStatement();
      CompoundStatement threadLoopBodyTwo = new CompoundStatement();
      List<Statement> children_before = beforeSync(sync);
      List<Statement> children_after = afterSync(sync);
      Statement lastChildAfter = null;
      
      for (Statement child : children_before) {
        child.detach();
        threadLoopBodyOne.addStatement(child);
      }

      for (Statement child : children_after) {
        child.detach();
        threadLoopBodyTwo.addStatement(child);
        lastChildAfter = child;
      }

      sync.detach();
      ThreadLoop threadLoopOne = null;

      if (threadLoopBodyOne.countStatements() != 0) {
        threadLoopOne = MCUDAUtils.NewNestedThreadLoop(numDims, threadLoopBodyOne);
        
        // **AP** Copy update/bound expressions from original threadLoop (Required in FCUDA)
        for (int dim = 0; dim < numDims; dim++) {
          threadLoopOne.setUpdate(dim, (Expression) outerLoopOne.getUpdate(dim).clone());
          threadLoopOne.setBound(dim, (Expression) outerLoopOne.getBound(dim).clone());
        }

        immediateParent.addStatement(threadLoopOne);
        immediateParent.addStatement(sync);
      } else if (!statementIsSync(sync))
        immediateParent.addStatement(sync);

      if (threadLoopBodyTwo.countStatements() != 0) {
        ThreadLoop threadLoopTwo = MCUDAUtils.NewNestedThreadLoop(numDims, threadLoopBodyTwo);

        // **AP** Copy update/bound expressions from original threadLoop (Required in FCUDA)
        
        for (int dim = 0; dim < numDims; dim++) {
          threadLoopTwo.setUpdate(dim, (Expression) outerLoopOne.getUpdate(dim).clone());
          threadLoopTwo.setBound(dim, (Expression) outerLoopOne.getBound(dim).clone());
        }

        immediateParent.addStatement(threadLoopTwo);

        if (!statementIsSync(lastChildAfter))
          immediateParent.addStatement(new ExpressionStatement(MCUDAUtils.getSync()));
      }

      if (immediateParent.getParent() instanceof CompoundStatement)
        syncOnStmt(immediateParent);
      else
        syncOnStmt((Statement)immediateParent.getParent());
    }
  }

  public void handleSyncs(Procedure proc)
  {
    PostOrderIterator iter = new PostOrderIterator(proc);
    FunctionCall syncTemplate = MCUDAUtils.getSync();
    boolean found_sync = false;

    while (iter.hasNext()) {
      FunctionCall sync;

      try {
        sync = (FunctionCall)iter.next(FunctionCall.class);
      } catch (NoSuchElementException e) {
        break;
      }

      if (sync.getName().compareTo(syncTemplate.getName()) == 0) {
        found_sync = true;
        //Danger here if the function call is not a simple
        //Expression Statement
        Statement syncStmt = (Statement)sync.getParent();
        syncOnStmt(syncStmt);
        //syncStmt.detach();
      }
    }

    if (false == found_sync)
      return;

    boolean madeChange = true;

    Set<Class<? extends Traversable>> set = new HashSet<Class<? extends Traversable>>();
    set.add(GotoStatement.class);
    set.add(SwitchStatement.class);
    set.add(BreakStatement.class);
    set.add(ContinueStatement.class);
    set.add(ReturnStatement.class);
    while (madeChange == true) {
      madeChange = false;
      iter = new PostOrderIterator(proc);
      Traversable o = null;

      while (iter.hasNext()) {
        try {
          o = iter.next(set);
        } catch (NoSuchElementException e) {
          break;
        }

        if (MCUDAUtils.getThreadLoopBody(o) == null)
          continue;

        if (o instanceof GotoStatement) {
          GotoStatement jump = (GotoStatement)o;
          Label target = jump.getTarget();

          if (MCUDAUtils.getThreadLoopBody(jump) !=
              MCUDAUtils.getThreadLoopBody(target)) {
            syncOnStmt(jump);
            syncOnStmt(target);
            madeChange = true;
              }

          continue;
        }

        if (o instanceof SwitchStatement &&
            MCUDAUtils.getThreadLoopBody(o) == null)
          //Not handled yet.
          continue;

        if (o instanceof BreakStatement) {
          BreakStatement breakstmt = (BreakStatement)o;
          Traversable structure = MCUDAUtils.getBreakContext(breakstmt);
          structure = structure.getParent();
          if (structure instanceof ThreadLoop) {
            syncOnStmt(breakstmt);
            madeChange = true;
            continue;
          }
        }

        if (o instanceof ContinueStatement) {
          Traversable structure = MCUDAUtils.getContinueContext(
              (ContinueStatement)o).getParent();
          if (structure instanceof ThreadLoop) {
            syncOnStmt((ContinueStatement)o);
            madeChange = true;
            continue;
          }
        }

        if (o instanceof ReturnStatement &&
            MCUDAUtils.getThreadLoopBody(o) != null) {
          ReturnStatement ret = (ReturnStatement)o;
          syncOnStmt(ret);
          madeChange = true;
          continue;
            }
      }
    }
  }


  public void transformProcedure(Procedure proc)
  {
    mProcedure = proc;
    List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc.getSymbolName());
    if (tskLst != null) {
      for (Procedure task : tskLst)
        if (FCUDAutils.getTaskType(task).equals("compute"))
          handleSyncs(task);
    }
  }
}
