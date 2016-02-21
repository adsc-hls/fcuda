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

import fcuda.common.*;
import fcuda.ir.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.analysis.DataFlow;
import cetus.exec.*;

/**
 *  Thread Unrolling
 *
 */
public class UnrollThreadLoops extends KernelTransformPass 
{
  private Procedure mProcedure;

  private boolean bconsUnroll;

  private int unrollFactor;

  private boolean bInLoop;

  private Expression loopIndex;

  private Expression loopBound;

  private Set<Expression> affectedIds;

  private Set<Expression> allAffectedIds;

  private LinkedList<VariableDeclaration> allDecl;

  private CompoundStatement newLoopbody;

  private LinkedList<IDExpression> unrollIndices;

  public enum StatementNames 
  {
    DefaultStatement,
    ExpressionStatement,
    WhileLoop,
    DeclarationStatement,
    CompoundStatement,
    AnnotationStatement,
    ForLoop,
    ThreadLoopStatement,
    IfStatement
  }

  public String getPassName() 
  {
    return new String("[UnrollThreadLoops-MCUDA]");
  }

  public UnrollThreadLoops(Program program) 
  {
    super(program);
  }

  public void resetUnroll() 
  {
    bInLoop     = false;
    affectedIds = null;

    allAffectedIds = new HashSet();

  }

  private StatementNames getStmtTypeId(Statement stmt) 
  {
    if (stmt instanceof WhileLoop)
      return StatementNames.WhileLoop;
    if (stmt instanceof ExpressionStatement)
      return StatementNames.ExpressionStatement;
    if (stmt instanceof CompoundStatement)
      return StatementNames.CompoundStatement;
    if (stmt instanceof DeclarationStatement)
      return StatementNames.DeclarationStatement;
    if (stmt instanceof AnnotationStatement)
      return StatementNames.AnnotationStatement;
    if (stmt instanceof ForLoop)
      return StatementNames.ForLoop;
    if (stmt instanceof ThreadLoop)
      return StatementNames.ThreadLoopStatement;
    if (stmt instanceof IfStatement)
      return StatementNames.IfStatement;
    return StatementNames.DefaultStatement;
  }

  public boolean isStepFormat(Expression expr) 
  {
    String str = expr.toString();
    if (str.indexOf('+') != -1)
      return true;
    return false;
  }

  public Expression createIndexPlusExpr(Expression index, Expression expr) 
  {
    Expression index0 = (Expression) index.clone();
    Expression new_expr = new BinaryExpression(index0, BinaryOperator.ADD, expr);
    return new_expr;
  }

  public Expression createIndexPlus(Expression index, int factor) 
  {
    Expression index0 = (Expression) index.clone();
    Expression add = new IntegerLiteral(factor);
    Expression expr = new BinaryExpression(index0, BinaryOperator.ADD, add);
    return expr;
  }

  public Expression createSuffixExpr(Expression index, int factor) 
  {
    if (index instanceof IDExpression) {
      IDExpression ide = (IDExpression) index;
      String name = ide.toString() + factor;
      IDExpression idExp = new NameID(name);
      VariableDeclarator idTor = new VariableDeclarator(idExp);
      Identifier id = new Identifier(idTor);
      return id;
    } else
      return index;
  }

  public boolean isObjectAffectedinAll(Object obj) 
  {
    if (affectedIds == null)
      return false;

    Iterator it = allAffectedIds.iterator();
    while (it.hasNext()) {
      Expression expr = (Expression) it.next();
      if (expr.equals(obj))
        return true;

      // special case loop index
      if (expr.equals(loopIndex)) {
        if (expr.toString().equals(obj.toString()))
          return true;
      }
    }
    return false;
  }

  public boolean isObjectAffected(Object obj) 
  {
    if (affectedIds == null)
      return false;

    Iterator it = affectedIds.iterator();
    while (it.hasNext()) {
      Expression expr = (Expression) it.next();
      if (expr.equals(obj))
        return true;
    }

    return false;
  }

  public boolean isStatementAffected(Statement cStmt) 
  {
    boolean bcopy = false;

    DepthFirstIterator iter = new DepthFirstIterator(cStmt);
    while (iter.hasNext()) {
      Object child = iter.next();
      if (isObjectAffected(child)) {
        bcopy = true;
        break;
      }
    }

    return bcopy;
  }

  public boolean isExpressionAffected(Expression expr) 
  {
    boolean bcopy = false;

    DepthFirstIterator iter = new DepthFirstIterator(expr);
    while (iter.hasNext()) {
      Object child = iter.next();
      if (isObjectAffected(child)) {
        bcopy = true;
        break;
      }
    }

    return bcopy;
  }

  public void updateAffectedStatement_consecutive(Statement cStmt,
      CompoundStatement newBody) 
  {

    /*
     * original statement, +1, +2, ,,, + unroll - 1
     */

    Procedure proc = (Procedure) IRTools.getAncestorOfType(cStmt, Procedure.class);

    for (int i = 0; i < unrollFactor; i++) {
      Statement newS = (Statement) cStmt.clone();
      List old_Exprs = new ArrayList();
      DepthFirstIterator newIter = new DepthFirstIterator(newS);
      while (newIter.hasNext()) {
        Object child = newIter.next();
        if (isObjectAffected(child)) {
          old_Exprs.add(child);
        }
      }

      for (Object old_expr : old_Exprs) {
        Expression new_expr = null;

        /*
         * Two cases 1. loopIndex: index + 0,1,2 2. other variables:
         * index012
         */
        if (old_expr.equals(loopIndex)) // i + 0
          new_expr = createIndexPlusExpr((Expression) old_expr,
              (Expression)unrollIndices.get(i).clone());
        else {
          new_expr = createSuffixExpr((Expression) old_expr, i);
          if (new_expr instanceof IDExpression)
            FCUDAGlobalData.addUnrolledID(proc.getName(), i, (IDExpression)new_expr);
        }
        IRTools.replaceAll((Traversable)old_expr, (Expression)old_expr, new_expr);
      }
      newBody.addStatement(newS);
    }

  }

  public void updateAffectedStatement_non_consecutive(Statement cStmt, CompoundStatement newBody) 
  {
    /*
     * original statement, +1, +2, ,,, + unroll - 1
     */

    System.out.println("Affected Statement: "+cStmt.toString());

    Procedure proc = (Procedure) IRTools.getAncestorOfType(cStmt, Procedure.class);

    for (int i = 0; i < unrollFactor; i++) {
      Statement newS = (Statement) cStmt.clone();
      List old_Exprs = new ArrayList();
      old_Exprs.clear();
      DepthFirstIterator newIter = new DepthFirstIterator(newS);
      while (newIter.hasNext()) {
        Object child = newIter.next();
        if (isObjectAffected(child)) {
          old_Exprs.add(child);
        }
      }

      for (Object old_expr : old_Exprs) {
        Expression new_expr = null;

        /*
         * Two cases 1. loopIndex: index + 0,1,2 2. other variables:
         * index012
         */
        if (old_expr.equals(loopIndex)) {
          System.out.println("[loopbound "+i+"]" + loopBound.toString());
          Expression offset1 = new BinaryExpression(
              (Expression)loopBound.clone(),
              BinaryOperator.DIVIDE,
              new IntegerLiteral(unrollFactor));
          Expression offset2 = new BinaryExpression(
              offset1,
              BinaryOperator.MULTIPLY,
              (Expression)unrollIndices.get(i).clone());
          new_expr = createIndexPlusExpr((Expression) old_expr, offset2);
        } else {
          new_expr = createSuffixExpr((Expression) old_expr, i);
          if (new_expr instanceof IDExpression)
            FCUDAGlobalData.addUnrolledID(proc.getName(), i, (IDExpression)new_expr);
        }

        System.out.println("\told expr: "+old_expr.toString()+"\n\tnew expr: "+new_expr.toString());
        IRTools.replaceAll((Traversable)old_expr, (Expression)old_expr, new_expr);
      }
      newBody.addStatement(newS);
    }

  }

  private void unrollDeclarationStatement(DeclarationStatement stmt, CompoundStatement newBody) 
  {
    System.out.println("[DeclarationStatement] " + stmt.toString());

    if (bInLoop)
      newBody.addANSIDeclaration(stmt.getDeclaration().clone());
  }

  private void unrollDefaultStatement(Statement stmt, CompoundStatement newBody) 
  {
    System.out.println("[DefaultStatement] " + stmt.toString());

    if (bInLoop) {
      if (isStatementAffected(stmt)) {
        // update the affected ids (most likely a new one).
        /* should be only one new definition */
        Set<Expression> defSet = DataFlow.mayDefine(stmt);
        Iterator it = defSet.iterator();
        while (it.hasNext()) {
          Expression expr = (Expression) it.next();

          if (expr instanceof ArrayAccess)
            continue;

          System.out.println("[def]" + expr.toString());
          affectedIds.add(expr);
        }

        if (bconsUnroll)
          updateAffectedStatement_consecutive(stmt, newBody);
        else
          updateAffectedStatement_non_consecutive(stmt, newBody);

      } else
        newBody.addStatement(stmt.clone());
    }
  }


  private void unrollThrCondExit(Statement stmt, CompoundStatement newBody) 
  {
    if (stmt instanceof IfStatement) {
      IfStatement ifStmt = (IfStatement) stmt;
      Expression origCond = ifStmt.getControlExpression();
      Expression newCond = null;
      int uIdx;

      for (uIdx = 0; uIdx < unrollFactor; uIdx++) {
        Expression unrolledCond = (Expression) origCond.clone();
        Set<Expression> condExprs = DataFlow.getUseSet(unrolledCond);
        System.out.println("Control use set: "+condExprs.toString());
        System.out.println("Affected Ids: "+affectedIds.toString());
        for (Expression expr : condExprs) {
          Expression newExpr = null;
          if (bconsUnroll) {
            // FILL THIS UP
          } else {
            if (expr.equals(loopIndex)) {
              Expression offset1 = new BinaryExpression(
                  (Expression)loopBound.clone(),
                  BinaryOperator.DIVIDE,
                  new IntegerLiteral(unrollFactor));
              Expression offset2 = new BinaryExpression(
                  offset1,
                  BinaryOperator.MULTIPLY,
                  (Expression)unrollIndices.get(uIdx).clone());
              newExpr = createIndexPlusExpr((Expression) expr, offset2);
            } else if (isExpressionAffected(expr)) {
              newExpr = createSuffixExpr((Expression) expr, uIdx);
            }

            if(newExpr != null)
              IRTools.replaceAll(expr, expr, newExpr);
          }
        }
        System.out.println("uIdx:"+uIdx+", unrolledCond: "+unrolledCond.toString());

        if (uIdx == 0)
          newCond = unrolledCond;
        else
          newCond = Symbolic.and(newCond, unrolledCond);

      } // Iterate through unroll indeces

      System.out.println("newCond: "+newCond.toString());
      IfStatement newIf = (IfStatement) ifStmt.clone();
      newIf.setControlExpression(newCond);

      newBody.addStatement(newIf);

    } else
      System.out.println("[WARNING]Statement's of type "+
          stmt.getClass().toString() + " are not handled (yet)!");

  }

  private void unrollForloopStatement(ForLoop stmt, CompoundStatement newBody) 
  {
    System.out.println("[ForloopStatement] " + stmt.toString());

    System.out.println("[Intitial]" + stmt.getInitialStatement());
    System.out.println("[Condition]" + stmt.getCondition());
    System.out.println("[Step]" + stmt.getStep());
    System.out.println("[Loopbody]" + stmt.getBody());

    Statement init = stmt.getInitialStatement();
    Expression cond = stmt.getCondition();
    Expression step = stmt.getStep();
    CompoundStatement loopbody = new CompoundStatement();
    ForLoop forloop = null;

    if ((isStatementAffected(init) ||
          isExpressionAffected(cond) ||
          isExpressionAffected(step)) && bInLoop) {

      if (bconsUnroll)
        updateAffectedStatement_consecutive(stmt, newBody);
      else
        updateAffectedStatement_non_consecutive(stmt, newBody);

    } else {
      forloop = new ForLoop(init.clone(), cond.clone(), step.clone(), loopbody);
      if (bInLoop)
        newBody.addStatement(forloop);
      unrollCompoundStatement((CompoundStatement) stmt.getBody(), loopbody);
    }
  }


  private void unrollIfStatement(IfStatement stmt, CompoundStatement newBody) {

    System.out.println("[IfStatement] " + stmt.toString());
    System.out.println("[Control] " + stmt.getControlExpression());

    Boolean belse = false;
    Expression control = stmt.getControlExpression();
    CompoundStatement thenStmt = new CompoundStatement();
    CompoundStatement elseStmt = null;

    if (stmt.getElseStatement() != null)
      belse = true;

    if (belse)
      elseStmt = new CompoundStatement();

    if (FCUDAGlobalData.isThrCondExit(stmt)) {
      if (bInLoop)
        unrollThrCondExit(stmt, newBody);
    } else if (isExpressionAffected(control)) {
      if(bInLoop) {
        if (bconsUnroll)
          updateAffectedStatement_consecutive(stmt, newBody);
        else
          updateAffectedStatement_non_consecutive(stmt, newBody);
      }
    } else {
      if (belse) {
        if (bInLoop)
          newBody.addStatement(new IfStatement(control.clone(), thenStmt, elseStmt));
        unrollCompoundStatement((CompoundStatement) stmt.getThenStatement(), thenStmt);
        unrollCompoundStatement((CompoundStatement) stmt.getElseStatement(), elseStmt);
      } else {
        if (bInLoop)
          newBody.addStatement(new IfStatement(control.clone(), thenStmt));
        unrollCompoundStatement((CompoundStatement) stmt.getThenStatement(), thenStmt);
      }
    }
  }

  private void unrollWhileLoopStatement(WhileLoop stmt,
      CompoundStatement newBody) 
  {
    System.out.println("[WhileloopStatement] " + stmt.toString());
    CompoundStatement whilebody = new CompoundStatement();

    WhileLoop whileloop = new WhileLoop(stmt.getCondition().clone(), whilebody);

    if (bInLoop)
      newBody.addStatement(whileloop);

    unrollCompoundStatement((CompoundStatement) stmt.getBody(), whilebody);
  }

  private void preUnrollThreadLoop() 
  {
    bInLoop = true;
    newLoopbody = null;
    newLoopbody = new CompoundStatement();

    affectedIds = null;
    affectedIds = new HashSet();
  }

  private void postUnrollThreadLoop() 
  {
    bInLoop = false;
    allAffectedIds.addAll(affectedIds);
  }

  private void unrollThreadLoopStatement(Statement cStmt) 
  {
    System.out.println("[Threadloop] " + cStmt.toString());
    int numDims = FCUDAGlobalData.getKernTblkDim(mProcedure);
    System.out.println("[numDims]" + numDims);

    preUnrollThreadLoop();

    ThreadLoop tloop = (ThreadLoop) cStmt;

    // update loop step
    loopIndex = MCUDAUtils.getTidID(numDims - 1);

    // update affected ids.
    affectedIds.add(loopIndex);

    loopBound = tloop.getBound(numDims - 1);
    System.out.println("[thread-loop loopbound]" + loopBound.toString());

    // update loop body
    unrollCompoundStatement((CompoundStatement) tloop.getBody(), newLoopbody);
    /*
     * TAN: This is quite a tricky situation. Why would we do unrolling two times?
     * For the first time, the Pass tries to identify potential variable which can
     * be unrolled (or duplicated) based on statement in which it is used. So it may
     * possibly lead to the case that the previous statement does not get unrolled.
     * That's why we need to do it two times here. The first time is just identifying
     * all potential affectIds and doing some preliminary unrolling. The second time
     * will ensure full unrolling.
     */
    newLoopbody = new CompoundStatement();
    unrollCompoundStatement((CompoundStatement) tloop.getBody(), newLoopbody);

    // unroll the outmost loop
    Expression expr = null;
    if (bconsUnroll) {
      expr = (Expression) new BinaryExpression(
          (Expression)loopIndex.clone(),
          BinaryOperator.ADD,
          new IntegerLiteral(unrollFactor));
      tloop.setUpdate(numDims - 1, expr);
    } else {
      expr = (Expression) new BinaryExpression(
          (Expression)loopBound.clone(),
          BinaryOperator.DIVIDE,
          new IntegerLiteral(unrollFactor));
      tloop.setBound(numDims - 1, expr);
    }

    // update variable declaration.
    insertVarDecl(newLoopbody);
    tloop.setBody(newLoopbody);

    postUnrollThreadLoop();
  }

  private void unrollCompoundStatement(CompoundStatement cStmt, CompoundStatement newBody) 
  {
    if (cStmt == null)
      return;

    FlatIterator flatIter = new FlatIterator(cStmt);

    while (flatIter.hasNext()) {
      Object currObj = flatIter.next();
      if (!(currObj instanceof Statement)) {
        System.out.println("Child " + currObj.toString()
            + " of compound statement is not statement");
        System.exit(0);
      }
      Statement currStmt = (Statement) currObj;


      switch (getStmtTypeId(currStmt)) {
        case DeclarationStatement:
          unrollDeclarationStatement((DeclarationStatement) currStmt, newBody);
          break;
        case ThreadLoopStatement:
          unrollThreadLoopStatement(currStmt);
          break;
        case ForLoop:
          unrollForloopStatement((ForLoop) currStmt, newBody);
          break;
        case IfStatement:
          unrollIfStatement((IfStatement) currStmt, newBody);
          break;
        case CompoundStatement:
          unrollCompoundStatement((CompoundStatement) currStmt, newBody);
          break;
        case WhileLoop:
          unrollWhileLoopStatement((WhileLoop) currStmt, newBody);
          break;
        case ExpressionStatement:
          unrollDefaultStatement(currStmt, newBody);
          break;
        default:
          unrollDefaultStatement(currStmt, newBody);
          break;
      }
    }

  }

  public void collectAllDeclaration(Procedure proc)
  {
    allDecl = new LinkedList();
    DepthFirstIterator diter = new DepthFirstIterator(proc.getBody());
    diter.pruneOn(Expression.class);
    while (diter.hasNext()) {
      Object child = diter.next();

      if (child instanceof VariableDeclaration)
        allDecl.add((VariableDeclaration) child);
    }

    System.out.println("[All Declaration]" + allDecl.toString());
  }


  public List<Specifier> getVariableType(IDExpression ide)
  {
    Iterator iter = allDecl.iterator();
    while(iter.hasNext()) {
      VariableDeclaration decl = (VariableDeclaration)iter.next();
      List<IDExpression>  list = decl.getDeclaredIDs();

      Iterator it = list.iterator();
      while(it.hasNext()) {
        IDExpression ide0 = (IDExpression) it.next();
        if(ide0.toString().equals(ide.toString())) {
          return decl.getSpecifiers();
        }
      }
    }
    List<Specifier> l= new ArrayList<Specifier>();
    l.add(Specifier.INT);
    return l;
  }

  private void createUnrollIdxDecls(Procedure proc)
  {
    int i;
    String unrollIdx = "uIdx";
    LinkedList<Specifier> idxSpecs = new LinkedList<Specifier>();
    idxSpecs.add(Specifier.CONST);
    idxSpecs.add(Specifier.INT);

    unrollIndices = new LinkedList<IDExpression>();
    unrollIndices.clear();

    for (i = 0; i< unrollFactor; i++) {
      IDExpression uIdxID = new NameID(unrollIdx + i);
      VariableDeclarator uIdxTor = new VariableDeclarator(uIdxID);
      Identifier uIdx = new Identifier(uIdxTor);
      unrollIndices.add(uIdx);
      System.out.println("--- Adding new ID to UnrolledIDs: "+uIdx.toString());
      FCUDAGlobalData.addUnrolledID(proc.getName(), i, uIdx);
      int iii = FCUDAGlobalData.isUnrolledID(proc.getName(), uIdx);
      System.out.println("------  new ID Unroll factor: "+iii);

      VariableDeclarator idxDeclor = new VariableDeclarator(uIdx);
      idxDeclor.setInitializer(new Initializer(new IntegerLiteral(i)));
      VariableDeclaration idxDecl = new VariableDeclaration(idxSpecs, idxDeclor);

      proc.getBody().addANSIDeclaration(idxDecl);
    }

  }


  public void unrollProc(Procedure proc) 
  {
    createUnrollIdxDecls(proc); // Create Constant declarations of unroll indices

    collectAllDeclaration(proc);

    unrollCompoundStatement(proc.getBody(), null);

    removeVarDecl(proc.getBody());
  }

  public void removeVarDecl(CompoundStatement procbody) 
  {
    /* remove the old ones
     * Assumptions: variable declarations are separate. int a; int b;
     */

    DepthFirstIterator diter = new DepthFirstIterator(procbody);
    while (diter.hasNext()) {
      Object child = diter.next();

      if (child instanceof DeclarationStatement) {
        DeclarationStatement decl = (DeclarationStatement) child;
        List<IDExpression> list = decl.getDeclaration().getDeclaredIDs();
        Iterator liter = list.iterator();
        while (liter.hasNext()) {
          IDExpression ide = (IDExpression) liter.next();
          if (isObjectAffectedinAll(ide) && ide != loopIndex) {
            System.out.println("[remove]" + ide.toString());
            decl.detach();
            break;
          }
        }
      }
    }
  }

  public void insertVarDecl(CompoundStatement procbody) 
  {
    for (Expression expr : affectedIds) {
      if (expr instanceof ArrayAccess)
        continue;
      if (expr.equals(loopIndex))
        continue;
      // only for name and identifier
      if (expr instanceof IDExpression) {
        System.out.println("[var]" + expr.toString());
        // check name conflicts.
        for (int i = 0; i < unrollFactor; i++) {
          Expression newID = createSuffixExpr(expr, i);
          Declarator declor = new VariableDeclarator((IDExpression) newID);
          Declaration decl = new VariableDeclaration(getVariableType((IDExpression)expr), declor);

          procbody.addDeclaration(decl);
        }
      }
    }
  }



  public void unroll(Procedure proc) 
  {
    System.out.println("\n[Unrolling] : " + proc.getName());
    System.out.println("[Proc]: " + proc.toString() + "\n");

    // Get Unroll factor
    unrollFactor = FCUDAutils.getTaskUnroll(proc);

    System.out.println("[unrollFactor] " + unrollFactor);

    if (unrollFactor > 1) {
      resetUnroll();
      unrollProc(proc);
    }
    return;
  }

  public void transformProcedure(Procedure proc) 
  {
    mProcedure = proc;
    bconsUnroll = false;
    List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc.getSymbolName());
    if (tskLst != null) {
      for (Procedure task : tskLst)
        if(FCUDAutils.getTaskType(task).equals("compute"))
          unroll(task);
    }
    FCUDAGlobalData.printUnrolledIDs();
  }
}
