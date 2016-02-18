package fcuda.transforms;

import java.util.*;

import fcuda.common.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.analysis.*;
import cetus.exec.*;
import cetus.transforms.*;

/*
 * Pass to split all if statements at the point where FCUDA pragma is
 * inserted so that AutoPilot can detect parallelism and also to achieve
 * communication and computation overlap
 */

public class IfSplitPass extends KernelTransformPass
{
  private Procedure mProcedure;
  private IDExpression mCurrEnableSignal;

  private HashSet<WhileLoop> mCreatedWhileLoops;
  private HashMap<WhileLoop, IDExpression> mCreatedWhileLoops2Guard;
  private HashMap<WhileLoop, IfStatement> mCreatedWhileLoops2IfStmt;

  private int mWhileLoopId;

  private Statement mCurrentFCUDACore;

  public IfSplitPass(Program program)
  {
    super(program);
    mCreatedWhileLoops = new HashSet();
    mCreatedWhileLoops2Guard = new HashMap();
    mCreatedWhileLoops2IfStmt = new HashMap();
    mWhileLoopId = 0;
    clear();
  }

  public void clear()
  {
    mCreatedWhileLoops.clear();
    mCreatedWhileLoops2Guard.clear();
    mCreatedWhileLoops2IfStmt.clear();
  }

  public String getPassName()
  {
    return new String("[IfSplitPass-FCUDA]");
  }

  // Extract task FunctionCall's from block-conditional if statements
  //     - handle the then/else clause of the block-conditional if which holds the task FunctionCall
  public Traversable handleIf(IfStatement ifStmt, Statement splitPos, Traversable prev)
  {
    Statement elseStmt = ifStmt.getElseStatement();
    Statement thenStmt = ifStmt.getThenStatement();
    Statement splitPosClone = splitPos.clone();

    boolean condFlag = true;
    Statement toSplit = null;

    //FIXME: for sure some of these checks will fail as more examples are used
    if (!splitPos.getParent().equals(prev)) {
      System.out.println("If stmt :\n"+ifStmt.toString()+"\n is not a parent of split position : "+splitPos.toString());
      System.exit(0);
    }
    if (elseStmt != null && prev.equals(elseStmt)) {
      condFlag = false;
      toSplit = elseStmt;
    }
    else {
      if(thenStmt.equals(prev))
      {
        condFlag = true;
        toSplit = thenStmt;
      }
      else {
        System.out.println("Parent of split position "+splitPos.toString()+" is neither if nor else part of "+ifStmt.toString());
        System.exit(0);
      }
    }
    if (!(toSplit instanceof CompoundStatement)) {
      System.out.println("Unsupported statement: "+toSplit.toString());
      System.exit(0);
    }
    CompoundStatement leadBlock = new CompoundStatement();
    List<Statement> leadStmts = new LinkedList<Statement>();
    CompoundStatement trailBlock = new CompoundStatement();
    List<Statement> trailStmts = new LinkedList<Statement>();

    boolean breakPoint = false;
    int leadCount = 0, trailCount = 0;
    FlatIterator codeIter = new FlatIterator(prev);
    while(codeIter.hasNext()) {
      Statement currStmt = null;
      try {
        currStmt = (Statement)codeIter.next(Statement.class);
      }
      catch (NoSuchElementException e) {
        break;
      }

      if (currStmt instanceof AnnotationStatement)
        continue;

      if (FCUDAutils.isSyncThreadStatement(currStmt))
        continue;

      if (currStmt.equals(splitPos))
        breakPoint = true;
      else {
        if (breakPoint) {
          trailStmts.add(currStmt);
          ++trailCount;
        } else {
          leadStmts.add(currStmt);
          leadCount++;
        }
      }
    }

    // Do actual statements transfers
    for (Statement tStmt : trailStmts) {
      tStmt.detach();
      trailBlock.addStatement(tStmt);
    }
    for (Statement lStmt : leadStmts) {
      lStmt.detach();
      // *TAN* check whether the statement is a DeclarationStatement
      //  If it is, add it to the main proc, not this If Block
      if (lStmt instanceof DeclarationStatement)
        mProcedure.getBody().addANSIDeclaration(((DeclarationStatement)lStmt).getDeclaration().clone());
      else
        leadBlock.addStatement(lStmt);
    }

    System.out.println("In if stmt, lead list "+leadBlock.toString());
    System.out.println("\nIn if stmt, trail list "+trailBlock.toString());

    Expression condExpr;
    condExpr = ifStmt.getControlExpression();
    if (!condFlag)
      condExpr = Symbolic.negate(condExpr);


    IfStatement leadIfStatement = null;
    if (leadCount > 0)
      if (!condFlag) {
        //split is in else part
        leadIfStatement = new IfStatement((Expression)(ifStmt.getControlExpression().clone()),
                              thenStmt.clone(), leadBlock);
        thenStmt.swapWith(leadIfStatement.getThenStatement());
      }
      else {
        if (elseStmt != null) {
          leadIfStatement = new IfStatement((Expression)condExpr.clone(), leadBlock, elseStmt.clone());
          elseStmt.swapWith(leadIfStatement.getElseStatement());
        } else
          leadIfStatement = new IfStatement((Expression)condExpr.clone(), leadBlock);
      }

    IfStatement trailIfStatement = null;
    if (trailCount > 0)
      if (!condFlag)
        trailIfStatement = new IfStatement((Expression)condExpr.clone(), trailBlock);
      else
        if (elseStmt != null) {
          trailIfStatement = new IfStatement((Expression)condExpr.clone(),
                                     trailBlock, elseStmt.clone());
          elseStmt.swapWith(trailIfStatement.getElseStatement());
        }
        else
          trailIfStatement = new IfStatement((Expression)condExpr.clone(),
              trailBlock);

    if (!(ifStmt.getParent() instanceof CompoundStatement)) {
      System.out.println("Unsupported statement: "+ ifStmt.getParent().toString());
      System.exit(0);
    }
    CompoundStatement parentStmt = (CompoundStatement)ifStmt.getParent();

    Statement lastStmt = ifStmt;

    if (leadIfStatement != null) {
      parentStmt.addStatementAfter(ifStmt, leadIfStatement);
      lastStmt = leadIfStatement;
    }

    if (!(splitPos instanceof WhileLoop && isCreatedByMe((WhileLoop)splitPos))) {
      if (splitPos.equals(mCurrentFCUDACore)) {
        if (!(mCurrentFCUDACore instanceof ExpressionStatement)
            || !(((ExpressionStatement)mCurrentFCUDACore).getExpression() instanceof FunctionCall)) {
         System.out.println("Unsupported split position: "+splitPos.toString());
         System.exit(0);
        }
        FunctionCall coreCall = ((FunctionCall)(
              ((ExpressionStatement)mCurrentFCUDACore)
              .getExpression()));
        Expression enableExpr = FCUDAGlobalData.getEnableExpression(coreCall);
        Expression binExpr = new BinaryExpression((Expression)(enableExpr.clone()),
            BinaryOperator.LOGICAL_AND,
            (Expression)condExpr.clone()
            );
        Tools.replaceAll(enableExpr.getParent(), enableExpr, binExpr);
        parentStmt.addStatementAfter(lastStmt, splitPosClone);
        splitPos.swapWith(splitPosClone);
      }
      else {
        ExpressionStatement exprStmt = new ExpressionStatement(
            new AssignmentExpression
            ((Expression)mCurrEnableSignal.clone(),
             AssignmentOperator.BITWISE_AND,
             (Expression)condExpr.clone()));
        parentStmt.addStatementAfter(lastStmt, exprStmt);
        parentStmt.addStatementAfter(exprStmt, splitPosClone);
        splitPos.swapWith(splitPosClone);
      }
    }
    else {
      parentStmt.addStatementAfter(lastStmt, splitPosClone);
      splitPos.swapWith(splitPosClone);
    }

    if(trailIfStatement != null)
      parentStmt.addStatementAfter(splitPos, trailIfStatement);

    ifStmt.detach();

    handleChildWhileLoops(ifStmt, condExpr);

    return splitPos;

  } // handleIf()

  public void handleChildWhileLoops(IfStatement ifStmt, Expression condExpr)
  {
    BreadthFirstIterator bfsIter = new BreadthFirstIterator(ifStmt);
    bfsIter.pruneOn(WhileLoop.class);
    CompoundStatement parentStmt = (CompoundStatement)(ifStmt.getParent());

    while(bfsIter.hasNext()) {
      WhileLoop currWhileLoop = null;
      try {
        currWhileLoop = (WhileLoop)bfsIter.next(WhileLoop.class);
      }
      catch(NoSuchElementException e) {
        break;
      }

      if(isCreatedByMe(currWhileLoop)) {
        ExpressionStatement condStmt = new ExpressionStatement(
            new AssignmentExpression(
              (IDExpression)(FCUDAGlobalData.getGuardVar(currWhileLoop).clone()),
              AssignmentOperator.BITWISE_AND,
              condExpr
              )
            );
        parentStmt.addStatementAfter(FCUDAGlobalData.getGuardInitStmt(currWhileLoop), condStmt);
      }
    }
  }

  // Convert loop into eternal while loop with:
  //     - initialization of block condition variable
  //     - break exit in case of no block condition variable being true
  //     - conditional execution of thread-independent statements (in terms of condition variable)
  public Statement handleFor(ForLoop forStmt, Statement splitPos)
  {
    if (!(forStmt.getParent() instanceof CompoundStatement)) {
      System.out.println("Unsupported statement: " + forStmt.getParent().toString());
      System.exit(0);
    }
    CompoundStatement parentStmt = (CompoundStatement)forStmt.getParent();

    Statement loopStmt = forStmt.getBody();
    CompoundStatement loopBody = null;
    CompoundStatement tmpCompound = new CompoundStatement();
    parentStmt.addStatementAfter(forStmt, tmpCompound);
    Statement splitPosClone = null;
    if (forStmt.getBody() instanceof CompoundStatement) {
      loopStmt.swapWith(tmpCompound);
      loopBody = (CompoundStatement)loopStmt;
    }
    else {
      splitPosClone = loopStmt.clone();  // create dummy clone for swapping
      tmpCompound.addStatement(splitPosClone);
      loopStmt.swapWith(splitPosClone); // *AP* do loopBody and splitPos refer to the same Statement always?
      loopBody = tmpCompound;
    }

    // Convert loop step expression to regular statement in loopBody
    Expression stepExpr = forStmt.getStep();
    if(stepExpr instanceof CommaExpression) {
      for(Traversable t : stepExpr.getChildren()) {
        ExpressionStatement stepStmt = new ExpressionStatement(((Expression)t).clone());
        loopBody.addStatement(stepStmt);
      }
    }
    else
      loopBody.addStatement(new ExpressionStatement(stepExpr.clone()));
    IDExpression whileLoopGuard = new NameID("whileLoopGuard_"+mWhileLoopId++);
    VariableDeclarator whileVDecl = new VariableDeclarator(whileLoopGuard);
    mProcedure.getBody().addANSIDeclaration(new VariableDeclaration(Specifier.INT, whileVDecl));

    Expression condExpr = forStmt.getCondition();
    AssignmentExpression whileLoopExpr = new AssignmentExpression(
        new Identifier(whileVDecl),
        AssignmentOperator.BITWISE_AND,
        condExpr.clone());
    ExpressionStatement condStmt = new ExpressionStatement(whileLoopExpr);

    ExpressionStatement guardInitStmt = new ExpressionStatement(
        new AssignmentExpression(
          new Identifier(whileVDecl),
          AssignmentOperator.NORMAL,
          new IntegerLiteral(1)));
    parentStmt.addStatementBefore(forStmt, guardInitStmt);

    IfStatement brkCondStmt = new IfStatement(Symbolic.negate(new Identifier(whileVDecl)),
        new BreakStatement());

    tmpCompound = new CompoundStatement();
    IfStatement ifGuard = new IfStatement(new Identifier(whileVDecl), tmpCompound);

    CompoundStatement whileLoopBody = new CompoundStatement();
    whileLoopBody.addStatement(condStmt);
    whileLoopBody.addStatementAfter(condStmt, brkCondStmt);
    whileLoopBody.addStatementAfter(brkCondStmt, ifGuard);

    WhileLoop whileStmt = new WhileLoop(new IntegerLiteral(1), whileLoopBody);

    FCUDAGlobalData.setGuardInitStmt(whileStmt, guardInitStmt);

    Statement initStmt = forStmt.getInitialStatement();
    if (initStmt instanceof ExpressionStatement &&
        ((ExpressionStatement)initStmt).getExpression() instanceof CommaExpression) {
      Statement curStmt  = null;
      Statement prevStmt = forStmt;
      CommaExpression ce = (CommaExpression)((ExpressionStatement)initStmt).getExpression();
      for (Traversable t : ce.getChildren()) {
        curStmt = new ExpressionStatement(((Expression)t).clone());
        parentStmt.addStatementAfter(prevStmt, curStmt);
        prevStmt = curStmt;
      }
      parentStmt.addStatementAfter(prevStmt, whileStmt);
    } else {
      parentStmt.addStatementAfter(forStmt, whileStmt);
      Statement initStmt_clone = initStmt.clone();
      parentStmt.addStatementAfter(forStmt, initStmt_clone);
      FCUDAGlobalData.setLoopVarInitStmt(whileStmt, initStmt_clone);
    }
    // Place loopBody in right AST position
    loopBody.swapWith(tmpCompound);

    FCUDAGlobalData.setWhileLoopCondition(whileStmt, whileLoopExpr);
    mCreatedWhileLoops.add(whileStmt);

    mCreatedWhileLoops2Guard.put(whileStmt, new Identifier(whileVDecl));
    mCreatedWhileLoops2IfStmt.put(whileStmt, brkCondStmt);

    forStmt.detach();
    tmpCompound.detach();

    handleIf(ifGuard, splitPos, loopBody);

    return whileStmt;
  } // handleFor()

  private boolean isCreatedByMe(WhileLoop t)
  {
    return mCreatedWhileLoops.contains(t);
  }

  public Traversable handleCreatedWhile(WhileLoop whileStmt)
  {
    IDExpression guard = mCreatedWhileLoops2Guard.get(whileStmt);
    IfStatement ifStmt = mCreatedWhileLoops2IfStmt.get(whileStmt);
    ExpressionStatement exprStmt = new ExpressionStatement(
        new AssignmentExpression
        ((Expression)mCurrEnableSignal.clone(),
         AssignmentOperator.BITWISE_AND,
         guard.clone()
        ));
    ((CompoundStatement)ifStmt.getParent()).addStatementAfter(ifStmt, exprStmt);
    return whileStmt;
  }

  public void handleCoveringControlFlow(Statement currStmt, Statement coreStmt)
  {
    Traversable prev = currStmt;
    Traversable parnt = currStmt.getParent();
    Statement splitPosition = coreStmt;
    while (parnt != mProcedure) {
      if (!(parnt instanceof Statement)) {
        System.out.println("Unsupported traversable: " + parnt.toString());
        System.exit(0);
      }
      if (parnt instanceof IfStatement)
        parnt = handleIf((IfStatement)parnt, splitPosition, prev);
      else {
        if (parnt instanceof ForLoop) {
          parnt = handleFor((ForLoop)parnt, splitPosition);
          splitPosition = (Statement)parnt;
        }
      }
      prev = parnt;
      parnt = parnt.getParent();
    }
  }

  public void runPass(Procedure proc)
  {
    mProcedure = proc;
    List<FunctionCall> fcudaCores = FCUDAGlobalData.getFcudaCores(mProcedure);

    System.out.println("fcudaCores:\n"+fcudaCores.toString());
    System.out.println("coreNames: \n"+FCUDAGlobalData.getCoreNames().toString());

    for (FunctionCall currCore : fcudaCores) {
      System.out.println("Handling control flow for "+currCore.toString());
      mCurrEnableSignal = FCUDAGlobalData.getEnableSignal(currCore);

      System.out.println("mCurrEnableSignal: "+mCurrEnableSignal.toString());

      Statement currStmt = FCUDAutils.getClosestParentStmt(currCore);
      mCurrentFCUDACore = currStmt;
      handleCoveringControlFlow(currStmt, currStmt);
    }
  }

  public void transformProcedure(Procedure proc)
  {
    if (!(FCUDAGlobalData.isConstMemProc(proc))) {
      runPass(proc);
      updateGlobalData();
    }
  }

  private void updateGlobalData()
  {
    for (WhileLoop currLoop : mCreatedWhileLoops) {
      IDExpression guardVar = mCreatedWhileLoops2Guard.get(currLoop);
      IfStatement guardIf = mCreatedWhileLoops2IfStmt.get(currLoop);
      FCUDAGlobalData.addWhileLoopData(currLoop, guardVar, guardIf);
    }
  }
}
